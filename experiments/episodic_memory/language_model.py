"""Independent Chinese intent frontend. No memory/answer labels are available here."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class IntentClassifier(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super().__init__()
        self.encoder = (
            AutoModel.from_pretrained(backbone, trust_remote_code=False)
            if pretrained
            else AutoModel.from_config(AutoConfig.from_pretrained(backbone))
        )
        self.head = nn.Linear(self.encoder.config.hidden_size, 6)

    def forward(self, inputs):
        return self.head(self.encoder(**inputs).last_hidden_state[:, 0])


def load_classifier(run, device):
    run = Path(run)
    model = IntentClassifier(run).to(device)
    model.load_state_dict(torch.load(run / "weights.pt", map_location=device, weights_only=True))
    model.eval()
    return (
        model,
        AutoTokenizer.from_pretrained(run),
        json.loads((run / "metadata.json").read_text()),
    )


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size=64):
    device = next(model.parameters()).device
    probabilities = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start : start + batch_size], padding=True, return_tensors="pt", truncation=False
        )
        if inputs["input_ids"].shape[1] > 128:
            raise ValueError("Do not silently truncate semantic questions")
        probabilities.append(model({k: v.to(device) for k, v in inputs.items()}).softmax(-1))
    return torch.cat(probabilities)
