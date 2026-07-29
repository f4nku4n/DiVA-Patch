import numpy as np
import random
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
        location_update_period=0,
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
        self.location_update_period = int(location_update_period)
        self.process = []
        self.best_patch = None
        self.best_image = None
        self.best_location = None
        self.best_success = None
        self.best_prediction = None
        self.best_l2 = None
        self.best_loss = None
        self.query_count = 0
        self.first_success_query = None

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
        if self.location_update_period < 0:
            raise ValueError("location_update_period must be non-negative")

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        self.image = image_tensor[None, :].to(device=device, dtype=torch.float32)
        self.image_normalized = (self.image - clip_min) / self.value_range

        _, _, height, width = self.image.shape
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        if patch_h <= 0 or patch_w <= 0 or patch_h > height or patch_w > width:
            raise ValueError("patch_size must fit inside the input image")

        self.patch_size = (patch_h, patch_w)
        self.image_size = (height, width)
        if location is None:
            location = self._sample_location()
        self._set_location(location)

    def _sample_location(self, exclude=None):
        height, width = self.image_size
        patch_h, patch_w = self.patch_size
        locations_h = height - patch_h + 1
        locations_w = width - patch_w + 1
        while True:
            location = (
                np.random.randint(0, locations_h),
                np.random.randint(0, locations_w),
            )
            if exclude is None or location != tuple(exclude):
                return location
            if locations_h * locations_w == 1:
                return location

    def _set_location(self, location):
        height, width = self.image_size
        patch_h, patch_w = self.patch_size
        x, y = int(location[0]), int(location[1])
        if x < 0 or y < 0 or x + patch_h > height or y + patch_w > width:
            raise ValueError("patch location is outside the input image")
        self.location = (x, y)
        self.mask = torch.zeros_like(self.image_normalized)
        self.mask[:, :, x:x + patch_h, y:y + patch_w] = 1.0

    def _should_update_location(self, iteration):
        return (
            self.location_update_period > 0
            and iteration > 1
            and (iteration - 1) % self.location_update_period == 0
        )

    def _normalized_patch(self, image_normalized):
        x, y = self.location
        patch_h, patch_w = self.patch_size
        return image_normalized[
            :, :, x:x + patch_h, y:y + patch_w
        ].detach().clone()

    def _place_normalized_patch(self, patch):
        image = self.image_normalized.clone()
        x, y = self.location
        patch_h, patch_w = self.patch_size
        image[:, :, x:x + patch_h, y:y + patch_w] = patch
        return image

    def _move_best_patch(self, project_eps=False):
        self._set_location(self._sample_location(exclude=self.location))
        image = self._place_normalized_patch(self.best_patch)
        if project_eps:
            delta = (image - self.image_normalized).clamp(-self.eps, self.eps)
            image = (self.image_normalized + delta * self.mask).clamp(0.0, 1.0)
        return image.detach()

    def _to_model_domain(self, image_normalized):
        return image_normalized * self.value_range + self.clip_min

    def _compose_model_input(self, image_normalized):
        patch_domain = self._to_model_domain(image_normalized)
        return self.image * (1.0 - self.mask) + patch_domain * self.mask

    def _logits(self, image_normalized):
        return self._forward_model(
            self._compose_model_input(image_normalized)
        )

    def _forward_model(self, image):
        logits = self.model.forward(image)
        self.query_count += 1
        prediction = int(logits.argmax(dim=1).item())
        if self._is_success(prediction) and self.first_success_query is None:
            self.first_success_query = self.query_count
        return logits

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
        l2 = self._l2(image_normalized)
        loss = float((-objective).item())
        if self._is_better(success, loss, l2):
            self.best_patch = self._normalized_patch(image_normalized)
            self.best_image = image_normalized.detach().clone()
            self.best_location = list(self.location)
            self.best_success = success
            self.best_prediction = prediction
            self.best_l2 = l2
            self.best_loss = loss
        self.process.append(
            [
                iteration,
                self.best_success,
                self.best_location.copy(),
                self._patch_from_best(),
                self.best_l2,
                self.best_loss,
            ]
        )
        return success, loss, l2

    def _is_better(self, success, loss, l2):
        if self.best_patch is None:
            return True
        if not self.best_success and success:
            return True
        if self.best_success and success:
            return l2 < self.best_l2
        if not self.best_success and not success:
            return loss < self.best_loss
        return False

    def _patch_from_best(self):
        patch = self.best_patch[0].permute(1, 2, 0)
        patch = patch * self.value_range + self.clip_min
        return patch.detach().cpu().numpy().copy()

    def _result(self):
        adversarial = self._compose_best_model_input()
        image = adversarial[0].permute(1, 2, 0).detach().cpu().numpy()
        return {
            "adversarial": self.best_success,
            "prediction": self.best_prediction,
            "location": self.best_location.copy(),
            "patch": self._patch_from_best(),
            "image": image,
            "l2": self.best_l2,
            "loss": self.best_loss,
            "queries": self.query_count,
            "first_success_query": self.first_success_query,
            "process": self.process,
        }

    def _compose_best_model_input(self):
        image = self.image.clone()
        x, y = self.best_location
        patch_h, patch_w = self.patch_size
        patch = self.best_patch * self.value_range + self.clip_min
        image[:, :, x:x + patch_h, y:y + patch_w] = patch
        return image


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
            if self._should_update_location(iteration):
                adversarial = self._move_best_patch(project_eps=True)
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

        return self._result()


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
            if self._should_update_location(iteration):
                adversarial = self._move_best_patch(project_eps=True)
                previous = adversarial.clone()
                best = adversarial.clone()
                best_objective = -torch.inf
                step_size = self.step_size
                checkpoint_objectives = []
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

        return self._result()


class LaVAN(WhiteBoxPatchAttack):
    def _final_objective(self, logits):
        return self._record_objective(logits)

    def _targeted_repo_loss(self, logits):
        source = torch.tensor([self.true_label], device=self.device)
        target = torch.tensor([self.target_label], device=self.device)
        return F.cross_entropy(logits, source) - F.cross_entropy(logits, target)

    def _record_objective(self, logits):
        if self.targeted:
            return -self._targeted_repo_loss(logits)
        return self._margin_objective(logits)

    def _initial_patch(self):
        patch_h, patch_w = self.patch_size
        patch = np.random.uniform(
            0.0, 1.0, (1, 3, patch_h, patch_w)
        ).astype(np.float32)
        return torch.from_numpy(patch).to(
            device=self.device,
            dtype=self.image_normalized.dtype,
        )

    def run(self):
        patch = self._initial_patch()
        adversarial = self._place_normalized_patch(patch).detach()
        adversarial.requires_grad_(True)
        logits = self._logits(adversarial)

        for iteration in tqdm(range(1, self.steps + 1)):
            if self._should_update_location(iteration):
                self._set_location(
                    self._sample_location(exclude=self.location)
                )
                patch = self.best_patch.detach().clone()
                adversarial = self._place_normalized_patch(patch).detach()
                adversarial.requires_grad_(True)
                logits = self._logits(adversarial)

            if self.targeted:
                update_loss = self._targeted_repo_loss(logits)
                gradient = torch.autograd.grad(update_loss, adversarial)[0]
                patch = patch - self.step_size * self._normalized_patch(
                    gradient
                )
            else:
                objective = self._margin_objective(logits)
                gradient = torch.autograd.grad(objective, adversarial)[0]
                patch = patch + self.step_size * self._normalized_patch(
                    gradient
                )

            patch = patch.clamp(0.0, 1.0).detach()
            adversarial = self._place_normalized_patch(patch).detach()
            adversarial.requires_grad_(True)
            logits = self._logits(adversarial)
            objective = self._record_objective(logits)
            self._record(iteration, adversarial, logits, objective)

        return self._result()


class LOAP(WhiteBoxPatchAttack):
    _DIRECTIONS = {
        "left": (0, -1),
        "right": (0, 1),
        "up": (-1, 0),
        "down": (1, 0),
    }

    def __init__(
        self,
        *args,
        lo_mode="full",
        stride=2,
        attempts=1,
        exclude_box=None,
        **kwargs,
    ):
        self.lo_mode = str(lo_mode)
        self.stride = int(stride)
        self.attempts = int(attempts)
        self.exclude_box = (
            None
            if exclude_box is None
            else tuple(int(value) for value in exclude_box)
        )
        self._allowed_locations_cache = None

        if self.lo_mode not in {"full", "random"}:
            raise ValueError("lo_mode must be either 'full' or 'random'")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.attempts <= 0:
            raise ValueError("attempts must be positive")
        if self.exclude_box is not None:
            if len(self.exclude_box) != 4:
                raise ValueError(
                    "exclude_box must contain top, left, height, and width"
                )
            top, left, height, width = self.exclude_box
            if top < 0 or left < 0 or height <= 0 or width <= 0:
                raise ValueError("exclude_box must describe a positive image region")

        super().__init__(*args, **kwargs)

        if self.exclude_box is not None:
            top, left, height, width = self.exclude_box
            image_h, image_w = self.image_size
            if top + height > image_h or left + width > image_w:
                raise ValueError("exclude_box must fit inside the input image")
        if not self._allowed_locations():
            raise ValueError("exclude_box leaves no valid patch location")

    def _location_is_allowed(self, location):
        row, column = int(location[0]), int(location[1])
        image_h, image_w = self.image_size
        patch_h, patch_w = self.patch_size
        if (
            row < 0
            or column < 0
            or row + patch_h > image_h
            or column + patch_w > image_w
        ):
            return False
        if self.exclude_box is None:
            return True

        top, left, height, width = self.exclude_box
        bottom = top + height
        right = left + width
        return (
            row + patch_h <= top
            or row >= bottom
            or column + patch_w <= left
            or column >= right
        )

    def _allowed_locations(self):
        if self._allowed_locations_cache is None:
            image_h, image_w = self.image_size
            patch_h, patch_w = self.patch_size
            self._allowed_locations_cache = [
                (row, column)
                for row in range(image_h - patch_h + 1)
                for column in range(image_w - patch_w + 1)
                if self._location_is_allowed((row, column))
            ]
        return self._allowed_locations_cache

    def _sample_location(self, exclude=None):
        locations = self._allowed_locations()
        if exclude is not None and len(locations) > 1:
            locations = [
                location
                for location in locations
                if location != tuple(exclude)
            ]
        if not locations:
            raise ValueError("no valid patch location is available")
        return random.choice(locations)

    def _set_location(self, location):
        super()._set_location(location)
        if not self._location_is_allowed(self.location):
            raise ValueError("patch location overlaps the excluded region")

    def _initial_patch(self):
        patch_h, patch_w = self.patch_size
        patch = np.random.uniform(
            0.0, 1.0, (1, 3, patch_h, patch_w)
        ).astype(np.float32)
        return torch.from_numpy(patch).to(
            device=self.device,
            dtype=self.image_normalized.dtype,
        )

    def _optimization_objective(self, logits):
        if self.targeted:
            target = torch.tensor([self.target_label], device=self.device)
            return -F.cross_entropy(logits, target)
        source = torch.tensor([self.true_label], device=self.device)
        return F.cross_entropy(logits, source)

    def _logits_for_patch(self, patch, location):
        row, column = location
        patch_h, patch_w = self.patch_size
        image = self.image.clone()
        image[:, :, row:row + patch_h, column:column + patch_w] = (
            patch * self.value_range + self.clip_min
        )
        return self._forward_model(image)

    def _candidate_location(self, location, direction):
        row_delta, column_delta = self._DIRECTIONS[direction]
        candidate = (
            location[0] + row_delta * self.stride,
            location[1] + column_delta * self.stride,
        )
        return candidate if self._location_is_allowed(candidate) else location

    def _optimize_location(self, patch):
        if self.lo_mode == "full":
            directions = tuple(self._DIRECTIONS)
        else:
            directions = (random.choice(tuple(self._DIRECTIONS)),)

        current_location = self.location
        with torch.no_grad():
            current_logits = self._logits_for_patch(patch, current_location)
            best_objective = float(
                self._optimization_objective(current_logits).item()
            )
            best_location = current_location
            for direction in directions:
                candidate = self._candidate_location(
                    current_location, direction
                )
                if candidate == current_location:
                    continue
                candidate_logits = self._logits_for_patch(patch, candidate)
                candidate_objective = float(
                    self._optimization_objective(candidate_logits).item()
                )
                if candidate_objective > best_objective:
                    best_objective = candidate_objective
                    best_location = candidate
        self._set_location(best_location)

    def run(self):
        global_iteration = 0
        for _ in range(self.attempts):
            self._set_location(self._sample_location())
            patch = self._initial_patch()

            for _ in tqdm(range(self.steps)):
                global_iteration += 1
                patch.requires_grad_(True)
                logits = self._logits_for_patch(patch, self.location)
                objective = self._optimization_objective(logits)
                gradient = torch.autograd.grad(objective, patch)[0]
                patch = (
                    patch + self.step_size * gradient.sign()
                ).clamp(0.0, 1.0).detach()

                self._optimize_location(patch)
                adversarial = self._place_normalized_patch(patch)
                with torch.no_grad():
                    logits = self._logits(adversarial)
                    objective = self._optimization_objective(logits)
                self._record(
                    global_iteration,
                    adversarial,
                    logits,
                    objective,
                )

        return self._result()
