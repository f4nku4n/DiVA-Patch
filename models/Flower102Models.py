"""Flowers102 victim model: EfficientNetV2-S fine-tuned on 102 classes."""
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from torchvision import models as torch_models


def preprocess_for_inference(tensor):
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"expected an NCHW RGB tensor, received shape {tuple(tensor.shape)}")

    # The Flowers checkpoint used Resize((256, 256)) followed by a 224 crop.
    tensor = F.resize(tensor, size=[256, 256], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    tensor = F.center_crop(tensor, [224, 224])
    return tensor.float() / 255.0


class Flower102Model:
    def __init__(self, device='cpu', setting='realistic'):
        # repo_id = "bengid/efficientnetv2-s-flower-classifier"
        # weights_path = hf_hub_download(
        #     repo_id=repo_id,
        #     filename="efficientnetv2-s-flower-classifier.safetensors",
        # )

        weights_path = 'model_weights/efficientnetv2-s-flower-classifier.safetensors'

        # root = '/kaggle/input/models/f4nku4n99/weights/pytorch/default/2'
        # weights_path = f'{root}/efficientnetv2-s-flower-classifier.safetensors'

        model = torch_models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 102)
        model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)

        self.setting = setting

        self.model = model.to(device)
        self.model.eval()

        self.mu = torch.tensor((0.4727, 0.3996, 0.3193), dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.sigma = torch.tensor((0.2965, 0.2471, 0.2812), dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def _logits(self, x):
        output = self.model((x - self.mu) / self.sigma)
        return output.logits if hasattr(output, "logits") else output

    @torch.inference_mode()
    def predict(self, x):
        if self.setting == 'realistic':
            x = preprocess_for_inference(x)
        return self._logits(x)

    @torch.inference_mode()
    def predict_many(self, images):
        if self.setting == 'realistic':
            images = [preprocess_for_inference(image) for image in images]
        else:
            images = list(images)
        if not images:
            raise ValueError("predict_many() requires at least one image")
        return self._logits(torch.cat(images, dim=0))

    def forward(self, x):
        if self.setting == 'realistic':
            x = preprocess_for_inference(x)
        return self._logits(x)

    def forward_many(self, images):
        """Differentiable heterogeneous-size forward used by white-box batches."""
        if self.setting == 'realistic':
            images = [preprocess_for_inference(image) for image in images]
        return self._logits(torch.cat(list(images), dim=0))

    def __call__(self, x):
        return self.predict(x)
