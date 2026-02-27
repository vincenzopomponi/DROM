import torch
import torch.nn.functional as F

# ---- paste your LanguageEncoder definition here ----
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LanguageEncoder(nn.Module):
    def __init__(self, output_dim=128, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        if model_name == "openai/clip-vit-base-patch32":
            hidden_size = self.encoder.config.text_config.hidden_size
        else:
            hidden_size = self.encoder.config.hidden_size

        # self.proj = nn.Sequential(
        #     nn.Linear(hidden_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, output_dim)
        # )

    def forward(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # 🔥 MOVE TOKENS TO MODEL DEVICE
        tokens = {k: v.to(self.encoder.device) for k, v in tokens.items()}

        embedding = self.encoder(**tokens)
        embedding = embedding.last_hidden_state.mean(dim=1)
        # embedding = self.proj(embedding)

        return embedding

    # def forward(self, texts):
    #     tokens = self.tokenizer(
    #         texts,
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt"
    #     )

    #     device = next(self.parameters()).device
    #     tokens = {k: v.to(device) for k, v in tokens.items()}

    #     outputs = self.encoder.get_text_features(**tokens)
    #     outputs = self.proj(outputs)
    #     embedding = torch.nn.functional.normalize(outputs, dim=-1)

    #     return embedding
# ---------------------------------------------------


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_name = "openai/clip-vit-base-patch32" # clip
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    encoder = LanguageEncoder(output_dim=384, model_name=model_name).to(device)
    encoder.eval()

    prompts = [
        "Grasp the red cube",
        "Grasp the blue cube",
        "Walk on top of mountain",
        "Open Drawer",
        "Pick Object",
        "Place Object",
        "Insert Fingers",
        "Close Drawer",
    ]

    with torch.no_grad():
        embeddings = encoder(prompts)
        embeddings = F.normalize(embeddings, dim=-1)

    print("\nCosine similarities:")

    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            sim = cosine_sim(embeddings[i], embeddings[j])
            print(f'"{prompts[i]}"  vs  "{prompts[j]}"  →  {sim:.3f}')


if __name__ == "__main__":
    main()
