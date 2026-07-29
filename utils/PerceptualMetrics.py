import numpy as np
import torch
from skimage.metrics import structural_similarity


def _validate_images(original, adversarial):
    original = np.asarray(original)
    adversarial = np.asarray(adversarial)
    if original.shape != adversarial.shape:
        raise ValueError("original and adversarial images must have the same shape")
    if original.ndim != 3 or original.shape[-1] != 3:
        raise ValueError("images must use HWC layout with three RGB channels")
    return original, adversarial


def _resolve_data_range(original, adversarial, data_range):
    if data_range is not None:
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        return float(data_range)
    maximum = max(float(np.max(original)), float(np.max(adversarial)))
    minimum = min(float(np.min(original)), float(np.min(adversarial)))
    return 1.0 if minimum >= 0.0 and maximum <= 1.0 else 255.0


def ssim_compute(original, adversarial, data_range=None):
    original, adversarial = _validate_images(original, adversarial)
    data_range = _resolve_data_range(original, adversarial, data_range)
    return float(
        structural_similarity(
            original.astype(np.float32),
            adversarial.astype(np.float32),
            data_range=data_range,
            channel_axis=-1,
        )
    )


class LPIPSAlexNet:
    def __init__(self, device="cpu"):
        try:
            import lpips
        except ImportError as error:
            raise ImportError(
                "LPIPS is not installed. Run `pip install lpips` or install "
                "the packages in requirement.txt."
            ) from error

        self.device = device
        self.model = lpips.LPIPS(net="alex", pretrained=True)
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _to_tensor(self, image, data_range):
        tensor = torch.from_numpy(
            np.ascontiguousarray(image, dtype=np.float32)
        ).permute(2, 0, 1)[None, :]
        tensor = tensor.to(self.device)
        tensor = (tensor / data_range).clamp(0.0, 1.0)
        return tensor * 2.0 - 1.0

    @torch.inference_mode()
    def __call__(self, original, adversarial, data_range=None):
        original, adversarial = _validate_images(original, adversarial)
        data_range = _resolve_data_range(original, adversarial, data_range)
        original_tensor = self._to_tensor(original, data_range)
        adversarial_tensor = self._to_tensor(adversarial, data_range)
        return float(self.model(original_tensor, adversarial_tensor).item())


class PerceptualMetrics:
    def __init__(self, device="cpu"):
        self.lpips = LPIPSAlexNet(device=device)

    def __call__(self, original, adversarial, data_range=None):
        return {
            "ssim": ssim_compute(
                original, adversarial, data_range=data_range
            ),
            "lpips": self.lpips(
                original, adversarial, data_range=data_range
            ),
        }


class PerceptualMetricTracker:
    def __init__(self, device, data_range):
        self.device = device
        self.data_range = data_range
        self.metrics = None
        self.ssim_scores = []
        self.lpips_scores = []

    @staticmethod
    def is_complete(summary):
        return "ssim" in summary and "lpips" in summary

    def add_summary(self, summary):
        self.ssim_scores.append(float(summary["ssim"]))
        self.lpips_scores.append(float(summary["lpips"]))

    def compute(self, original, adversarial):
        if self.metrics is None:
            self.metrics = PerceptualMetrics(device=self.device)
        summary = self.metrics(
            original, adversarial, data_range=self.data_range
        )
        self.add_summary(summary)
        return summary

    def print_summary(self):
        if not self.ssim_scores:
            return
        print(f"SSIM (mean): {np.mean(self.ssim_scores):.4f}")
        print(f"LPIPS-AlexNet (mean): {np.mean(self.lpips_scores):.4f}")
