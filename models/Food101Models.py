"""Food-101 victim model: Swin fine-tuned on 101 classes."""
import torch
import torchvision.transforms.functional as F

from transformers import AutoImageProcessor, AutoModelForImageClassification

def preprocess_for_inference(tensor):
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"expected an NCHW RGB tensor, received shape {tuple(tensor.shape)}")

    tensor = F.resize(tensor, size=[224, 224], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    return tensor.float() / 255.0

class Food101Model:
    def __init__(self, device='cpu', setting='realistic'):
        processor = AutoImageProcessor.from_pretrained('model_weights/swin-finetuned-food101', local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained('model_weights/swin-finetuned-food101', local_files_only=True)

        # root = '/kaggle/input/models/f4nku4n99/weights/pytorch/default/2'
        # processor = AutoImageProcessor.from_pretrained(f'{root}/swin-finetuned-food101', local_files_only=True)
        # model = AutoModelForImageClassification.from_pretrained(f'{root}/swin-finetuned-food101', local_files_only=True)

        mean, std = tuple(processor.image_mean), tuple(processor.image_std)

        self.setting = setting

        self.model = model.to(device)
        self.model.eval()

        self.mu = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.sigma = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

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
