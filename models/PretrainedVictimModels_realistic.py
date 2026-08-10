"""Realistic-input wrappers for the three dataset-specific victim models.

Inputs are float NCHW image tensors in [0, 255].  Preprocessing stays in
PyTorch so ``forward`` remains differentiable for white-box attacks.
"""

import torch
import torchvision.transforms.functional as F

from models.PretrainedVictimModels import PretrainedVictimModel


def preprocess_for_inference(tensor, model_index):
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"expected an NCHW RGB tensor, received shape {tuple(tensor.shape)}")

    # The Flowers checkpoint used Resize((256, 256)) followed by a 224 crop.
    if model_index == 0:
        tensor = F.resize(
            tensor,
            size=[256, 256],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = F.center_crop(tensor, [224, 224])
    else:
        # ViTImageProcessor and SwinImageProcessor resize to a 224 square.
        tensor = F.resize(
            tensor,
            size=[224, 224],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )
    return tensor.float() / 255.0


class RealisticPretrainedVictimModel(PretrainedVictimModel):
    def _logits(self, x):
        x = preprocess_for_inference(x, self.model_index)
        return super()._logits(x)
