import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


# ============================================================
# 1. CLIP-based Language Encoder
# ============================================================

class CLIPLanguageEncoder(nn.Module):
    def __init__(self, output_dim=128, model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = CLIPModel.from_pretrained(model_name)

        clip_dim = self.encoder.config.projection_dim  # 512 for ViT-B/32

        self.proj = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # CLIP text features are already pooled
        text_features = self.encoder.get_text_features(**tokens)

        embedding = self.proj(text_features)
        embedding = F.normalize(embedding, dim=-1)

        return embedding


# ============================================================
# 2. Utility functions
# ============================================================

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def encode_texts(encoder, texts):
    encoder.eval()
    with torch.no_grad():
        emb = encoder(texts)
        emb = F.normalize(emb, dim=-1)
    return emb


# ============================================================
# 3. Relative similarity (ranking-based) evaluation
# ============================================================

def relative_similarity_accuracy(encoder, dataset):
    correct = 0
    total = len(dataset)

    print("\n========== Relative Similarity Evaluation ==========")

    for idx, sample in enumerate(dataset):
        texts = (
            [sample["anchor"], sample["positive"]] +
            sample["negatives"]
        )

        embeddings = encode_texts(encoder, texts)

        anchor_emb = embeddings[0]
        pos_emb = embeddings[1]
        neg_embs = embeddings[2:]

        pos_sim = cosine_sim(anchor_emb, pos_emb)
        neg_sims = [cosine_sim(anchor_emb, n) for n in neg_embs]

        is_correct = pos_sim > max(neg_sims)
        correct += int(is_correct)

        print(f"\nSample {idx + 1}")
        print(f"Anchor:   {sample['anchor']}")
        print(f"Positive: {sample['positive']} → sim = {pos_sim:.3f}")

        for neg, s in zip(sample["negatives"], neg_sims):
            print(f"Negative: {neg:30s} → sim = {s:.3f}")

        print(f"Result: {'✔ correct' if is_correct else '✘ incorrect'}")

    accuracy = correct / total
    print("\n===================================================")
    print(f"Relative similarity accuracy: {accuracy:.3f}")

    return accuracy


# ============================================================
# 4. Example evaluation dataset
# ============================================================

dataset = [
    {
        "anchor": "Grasp the red cube",
        "positive": "Pick up the blue cube",
        "negatives": [
            "Walk on top of mountain",
            "Open the drawer",
            "Move to the table"
        ]
    },
    {
        "anchor": "Open the drawer",
        "positive": "Pull the drawer open",
        "negatives": [
            "Grasp the cube",
            "Walk on top of mountain"
        ]
    },
    {
        "anchor": "Move to the table",
        "positive": "Go to the table",
        "negatives": [
            "Open the drawer",
            "Pick up the cube"
        ]
    }
]


# ============================================================
# 5. Main
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = CLIPLanguageEncoder(output_dim=128)
    encoder = encoder.to(device)

    # Freeze CLIP backbone (recommended for evaluation)
    for p in encoder.encoder.parameters():
        p.requires_grad = False

    relative_similarity_accuracy(encoder, dataset)


if __name__ == "__main__":
    main()
