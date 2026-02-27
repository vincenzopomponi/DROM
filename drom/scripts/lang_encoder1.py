"""
Language Encoder with Transformer Head and Contrastive Training

This script implements a task-oriented language encoder designed to produce
embeddings where prompts referring to the same task have high cosine similarity.

Architecture:
- Frozen pretrained language model (MiniLM)
- Trainable Transformer encoder on top of token embeddings
- Projection head producing L2-normalized task embeddings

Training:
- Contrastive (InfoNCE) loss on pairs of prompts describing the same task
- Encourages task-level invariance (e.g., color, wording) and separation
  from unrelated tasks

Evaluation:
- Cosine similarity between embeddings
- Relative similarity: same-task prompts rank higher than different tasks

Intended use:
- Language-conditioned robotics and manipulation
- Task / subtask embedding learning
- Conditioning policies or planners on natural language goals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from lang.data import generate_training_pairs

class LanguageEncoder(nn.Module):
    def __init__(
        self,
        output_dim=128,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        n_transformer_layers=2,
        n_heads=4,
        dim_ff=512,
        dropout=0.1,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        # 🔒 Freeze pretrained LM
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 🔥 Trainable transformer on top of LM tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            lm_out = self.encoder(**tokens)

        # Token-level embeddings
        x = lm_out.last_hidden_state  # [B, T, H]

        # Attention mask → transformer mask
        attn_mask = tokens["attention_mask"] == 0

        # Transformer reasoning over tokens
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Mean pooling (task-level embedding)
        pooled = x.masked_fill(attn_mask.unsqueeze(-1), 0.0).sum(dim=1)
        pooled = pooled / (~attn_mask).sum(dim=1, keepdim=True)

        embedding = self.proj(pooled)
        embedding = F.normalize(embedding, dim=-1)

        return embedding

def contrastive_loss(embeddings, temperature=0.07):
    """
    embeddings: [2B, D]
    first B = anchors, second B = positives
    """
    sim = embeddings @ embeddings.T / temperature
    B = embeddings.size(0) // 2

    labels = torch.arange(B, device=embeddings.device)
    labels = torch.cat([labels + B, labels], dim=0)

    mask = torch.eye(2 * B, device=embeddings.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    return F.cross_entropy(sim, labels)

def train_encoder(encoder, train_pairs, epochs=100, lr=1e-3, batch_size=4):
    device = next(encoder.parameters()).device
    optimizer = torch.optim.AdamW(
        list(encoder.transformer.parameters()) +
        list(encoder.proj.parameters()),
        lr=lr,
    )

    encoder.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i + batch_size]
            anchors, positives = zip(*batch)

            texts = list(anchors) + list(positives)
            embeddings = encoder(texts)

            loss = contrastive_loss(embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f}")

    encoder.eval()

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def evaluate(encoder):
    prompts = [
        "Grasp the red cube",
        "Grasp the blue cube",
        "Walk on top of mountain",
    ]

    with torch.no_grad():
        emb = encoder(prompts)

    print("\nCosine similarities:")
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            sim = cosine_sim(emb[i], emb[j])
            print(f'"{prompts[i]}" vs "{prompts[j]}" → {sim:.3f}')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = LanguageEncoder(output_dim=128).to(device)

    train_pairs = generate_training_pairs()

    print("\n--- Training transformer-based encoder ---")
    train_encoder(encoder, train_pairs)

    print("\n--- Evaluation ---")
    evaluate(encoder)


if __name__ == "__main__":
    main()
