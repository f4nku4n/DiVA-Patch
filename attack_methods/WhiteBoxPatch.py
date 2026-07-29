import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class WhiteBoxPatchAttack:
    def __init__(
        self,
        image,
        model,
        true_label,
        target_label=None,
        targeted=False,
        patch_size=(40, 40),
        steps=100,
        step_size=0.01,
        eps=1.0,
        clip_min=0.0,
        clip_max=1.0,
        device="cuda",
        location=None,
    ):
        self.model = model
        self.true_label = int(true_label)
        self.target_label = None if target_label is None else int(target_label)
        self.targeted = targeted
        self.steps = steps
        self.step_size = step_size
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.value_range = clip_max - clip_min
        self.device = device
        self.process = []

        if targeted and target_label is None:
            raise ValueError("target_label is required for a targeted attack")
        if self.value_range <= 0:
            raise ValueError("clip_max must be greater than clip_min")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if eps < 0:
            raise ValueError("eps must be non-negative")

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        self.image = image_tensor[None, :].to(device=device, dtype=torch.float32)
        self.image_normalized = (self.image - clip_min) / self.value_range

        _, _, height, width = self.image.shape
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        if patch_h <= 0 or patch_w <= 0 or patch_h > height or patch_w > width:
            raise ValueError("patch_size must fit inside the input image")

        if location is None:
            x = np.random.randint(0, height - patch_h + 1)
            y = np.random.randint(0, width - patch_w + 1)
            location = (x, y)
        self.location = (int(location[0]), int(location[1]))
        x, y = self.location
        if x < 0 or y < 0 or x + patch_h > height or y + patch_w > width:
            raise ValueError("patch location is outside the input image")

        self.patch_size = (patch_h, patch_w)
        self.mask = torch.zeros_like(self.image_normalized)
        self.mask[:, :, x:x + patch_h, y:y + patch_w] = 1.0

    def _to_model_domain(self, image_normalized):
        return image_normalized * self.value_range + self.clip_min

    def _compose_model_input(self, image_normalized):
        patch_domain = self._to_model_domain(image_normalized)
        return self.image * (1.0 - self.mask) + patch_domain * self.mask

    def _logits(self, image_normalized):
        return self.model.forward(self._compose_model_input(image_normalized))

    def _ce_objective(self, logits):
        if self.targeted:
            label = torch.tensor([self.target_label], device=self.device)
            return -F.cross_entropy(logits, label)
        label = torch.tensor([self.true_label], device=self.device)
        return F.cross_entropy(logits, label)

    def _margin_objective(self, logits):
        if self.targeted:
            return logits[0, self.target_label] - logits[0, self.true_label]
        other_logits = logits[0].clone()
        other_logits[self.true_label] = -torch.inf
        return other_logits.max() - logits[0, self.true_label]

    def _final_objective(self, logits):
        return self._ce_objective(logits)

    def _is_success(self, prediction):
        if self.targeted:
            return prediction == self.target_label
        return prediction != self.true_label

    def _patch_from_image(self, image_normalized):
        x, y = self.location
        patch_h, patch_w = self.patch_size
        patch = self._compose_model_input(image_normalized)[
            0, :, x:x + patch_h, y:y + patch_w
        ]
        return patch.permute(1, 2, 0).detach().cpu().numpy()

    def _l2(self, image_normalized):
        difference = (image_normalized - self.image_normalized) * self.mask
        return float(torch.linalg.vector_norm(difference).item())

    def _record(self, iteration, image_normalized, logits, objective):
        prediction = int(logits.argmax(dim=1).item())
        success = self._is_success(prediction)
        patch = self._patch_from_image(image_normalized)
        l2 = self._l2(image_normalized)
        loss = float((-objective).item())
        self.process.append(
            [iteration, success, list(self.location), patch.copy(), l2, loss]
        )
        return success, loss, l2

    def _result(self, image_normalized):
        logits = self._logits(image_normalized)
        objective = self._final_objective(logits)
        prediction = int(logits.argmax(dim=1).item())
        adversarial = self._compose_model_input(image_normalized)
        image = adversarial[0].permute(1, 2, 0).detach().cpu().numpy()
        return {
            "adversarial": self._is_success(prediction),
            "prediction": prediction,
            "location": list(self.location),
            "patch": self._patch_from_image(image_normalized),
            "image": image,
            "l2": self._l2(image_normalized),
            "loss": float((-objective).item()),
            "process": self.process,
        }


class MaskedPGD(WhiteBoxPatchAttack):
    def run(self):
        delta = torch.empty_like(self.image_normalized).uniform_(
            -self.eps, self.eps
        )
        delta = delta * self.mask
        adversarial = (self.image_normalized + delta).clamp(0.0, 1.0)
        adversarial = (
            self.image_normalized * (1.0 - self.mask) + adversarial * self.mask
        ).detach()

        for iteration in tqdm(range(1, self.steps + 1)):
            adversarial.requires_grad_(True)
            logits = self._logits(adversarial)
            objective = self._ce_objective(logits)
            self._record(iteration, adversarial, logits, objective)
            if iteration == self.steps:
                adversarial = adversarial.detach()
                continue

            gradient = torch.autograd.grad(objective, adversarial)[0]
            candidate = adversarial + self.step_size * gradient.sign() * self.mask
            delta = (candidate - self.image_normalized).clamp(
                -self.eps, self.eps
            )
            adversarial = (self.image_normalized + delta * self.mask).clamp(
                0.0, 1.0
            ).detach()

        return self._result(adversarial)


class MaskedAutoPGD(WhiteBoxPatchAttack):
    def run(self):
        delta = torch.empty_like(self.image_normalized).uniform_(
            -self.eps, self.eps
        )
        delta = delta * self.mask
        adversarial = (self.image_normalized + delta).clamp(0.0, 1.0)
        adversarial = (
            self.image_normalized * (1.0 - self.mask) + adversarial * self.mask
        ).detach()

        previous = adversarial.clone()
        best = adversarial.clone()
        best_objective = -torch.inf
        step_size = self.step_size
        checkpoint = max(self.steps // 10, 1)
        checkpoint_objectives = []

        for iteration in tqdm(range(1, self.steps + 1)):
            adversarial.requires_grad_(True)
            logits = self._logits(adversarial)
            objective = self._ce_objective(logits)
            self._record(iteration, adversarial, logits, objective)

            objective_value = float(objective.item())
            checkpoint_objectives.append(objective_value)
            if objective_value > float(best_objective):
                best_objective = objective.detach()
                best = adversarial.detach().clone()

            if iteration == self.steps:
                adversarial = adversarial.detach()
                continue

            gradient = torch.autograd.grad(objective, adversarial)[0]
            projected = adversarial + step_size * gradient.sign() * self.mask
            delta = (projected - self.image_normalized).clamp(
                -self.eps, self.eps
            )
            projected = (self.image_normalized + delta * self.mask).clamp(
                0.0, 1.0
            )

            momentum = 0.75 if iteration > 1 else 1.0
            candidate = adversarial + momentum * (projected - adversarial)
            candidate += (1.0 - momentum) * (adversarial - previous)
            delta = (candidate - self.image_normalized).clamp(
                -self.eps, self.eps
            )
            candidate = (self.image_normalized + delta * self.mask).clamp(
                0.0, 1.0
            )
            previous = adversarial.detach()
            adversarial = candidate.detach()

            if iteration % checkpoint == 0:
                improvements = sum(
                    current > prior
                    for prior, current in zip(
                        checkpoint_objectives, checkpoint_objectives[1:]
                    )
                )
                if improvements <= len(checkpoint_objectives) // 2:
                    step_size *= 0.5
                    adversarial = best.clone()
                    previous = best.clone()
                checkpoint_objectives = []

        return self._result(best)


class LaVAN(WhiteBoxPatchAttack):
    def _final_objective(self, logits):
        return self._margin_objective(logits)

    def run(self):
        x, y = self.location
        patch_h, patch_w = self.patch_size
        patch = torch.zeros(
            (1, 3, patch_h, patch_w),
            device=self.device,
            dtype=self.image_normalized.dtype,
        )

        adversarial = self.image_normalized.clone()
        adversarial[:, :, x:x + patch_h, y:y + patch_w] = patch
        adversarial = adversarial.detach()

        for iteration in tqdm(range(1, self.steps + 1)):
            adversarial.requires_grad_(True)
            logits = self._logits(adversarial)
            objective = self._margin_objective(logits)
            self._record(iteration, adversarial, logits, objective)
            if iteration == self.steps:
                adversarial = adversarial.detach()
                continue

            gradient = torch.autograd.grad(objective, adversarial)[0]
            adversarial = (
                adversarial + self.step_size * gradient * self.mask
            ).clamp(0.0, 1.0)
            adversarial = (
                self.image_normalized * (1.0 - self.mask)
                + adversarial * self.mask
            ).detach()

        return self._result(adversarial)
