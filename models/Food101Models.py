"""Food-101 victim model: Swin fine-tuned on 101 classes."""

from models.PretrainedVictimModels import PretrainedVictimModel
from models.PretrainedVictimModels_realistic import RealisticPretrainedVictimModel


class Food101Model(PretrainedVictimModel):
    """Accept NCHW float images in [0, 1] with spatial size 224 x 224."""

    def __init__(self, device="cpu"):
        super().__init__(2, device)


class Food101ModelRealistic(RealisticPretrainedVictimModel):
    """Accept raw NCHW float images in [0, 255]."""

    def __init__(self, device="cpu"):
        super().__init__(2, device)
