"""Flowers102 victim model: EfficientNetV2-S fine-tuned on 102 classes."""

from models.PretrainedVictimModels import PretrainedVictimModel
from models.PretrainedVictimModels_realistic import RealisticPretrainedVictimModel


class Flower102Model(PretrainedVictimModel):
    """Accept NCHW float images in [0, 1] with spatial size 224 x 224."""

    def __init__(self, device="cpu"):
        super().__init__(0, device)


class Flower102ModelRealistic(RealisticPretrainedVictimModel):
    """Accept raw NCHW float images in [0, 255]."""

    def __init__(self, device="cpu"):
        super().__init__(0, device)
