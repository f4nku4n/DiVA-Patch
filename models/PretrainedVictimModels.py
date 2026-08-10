"""Wrappers for the three dataset-specific pretrained victim models.

Model indices:
    0: Flowers102 EfficientNetV2-S
    1: Stanford Cars ViT-B/16
    2: Food-101 Swin

The common wrapper expects float NCHW tensors in [0, 1].  Input tensors must
already have the spatial size expected by the selected model (224 x 224 for
the three current checkpoints).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import models as torch_models
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAMES = (
    "Flowers102-EfficientNetV2-S",
    "StanfordCars-ViT-B16",
    "Food101-Swin",
)

_FLOWERS_MEAN = (0.4727, 0.3996, 0.3193)
_FLOWERS_STD = (0.2965, 0.2471, 0.2812)
_HF_MODEL_IDS = {
    1: "therealcyberlord/stanford-car-vit-patch16",
    2: "aspis/swin-finetuned-food101",
}


def _build_model(model_index: int):
    """Return ``(model, mean, std)`` for a victim index."""
    if model_index == 0:
        repo_id = "bengid/efficientnetv2-s-flower-classifier"
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="efficientnetv2-s-flower-classifier.safetensors",
        )
        model = torch_models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 102)
        model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
        return model, _FLOWERS_MEAN, _FLOWERS_STD

    if model_index in _HF_MODEL_IDS:
        model_id = _HF_MODEL_IDS[model_index]
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        return model, tuple(processor.image_mean), tuple(processor.image_std)

    raise ValueError(f"model must be 0, 1, or 2; received {model_index!r}")


class PretrainedVictimModel:
    """Common wrapper for one of the dataset-specific victim models."""

    def __init__(self, model: int, device="cpu"):
        self.model_index = model
        self.device = torch.device(device)
        model_pt, mean, std = _build_model(model)
        self.model = model_pt.to(self.device).eval()
        self.mu = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.sigma = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _logits(self, x):
        output = self.model((x - self.mu) / self.sigma)
        return output.logits if hasattr(output, "logits") else output

    @torch.inference_mode()
    def predict(self, x):
        return self._logits(x)

    def forward(self, x):
        return self._logits(x)

    def __call__(self, x):
        return self.predict(x)


