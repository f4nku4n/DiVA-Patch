import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class BatchedWhiteBoxPatchAttack:
    def __init__(
        self,
        images,
        model,
        true_labels,
        target_labels=None,
        targeted=False,
        patch_size=(40, 40),
        steps=100,
        step_size=0.01,
        eps=1.0,
        device="cuda",
        location_update_period=0,
        early_stop=False,
        **_,
    ):
        self.model = model
        self.device = device
        self.targeted = bool(targeted)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.location_update_period = int(location_update_period)
        self.early_stop = bool(early_stop)
        self.images = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2)
        self.images = self.images.to(device=device, dtype=torch.float32)
        self.true_labels = torch.as_tensor(
            true_labels, device=device, dtype=torch.long
        )
        self.target_labels = (
            None
            if target_labels is None
            else torch.as_tensor(
                target_labels, device=device, dtype=torch.long
            )
        )
        self.batch_size, _, self.height, self.width = self.images.shape
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        ph, pw = self.patch_size
        if ph <= 0 or pw <= 0 or ph > self.height or pw > self.width:
            raise ValueError("patch_size must fit inside the input images")
        if self.targeted and self.target_labels is None:
            raise ValueError("target_labels are required for targeted attacks")

        self.locations = [self._sample_location() for _ in range(self.batch_size)]
        self.best_patch = [None] * self.batch_size
        self.best_location = [None] * self.batch_size
        self.best_success = [None] * self.batch_size
        self.best_prediction = [None] * self.batch_size
        self.best_l2 = [None] * self.batch_size
        self.best_loss = [None] * self.batch_size
        self.process = [[] for _ in range(self.batch_size)]
        self.query_count = [0] * self.batch_size
        self.first_success_query = [None] * self.batch_size

    def _sample_location(self, exclude=None):
        ph, pw = self.patch_size
        while True:
            location = (
                np.random.randint(0, self.height - ph + 1),
                np.random.randint(0, self.width - pw + 1),
            )
            if exclude is None or location != tuple(exclude):
                return location
            if (self.height - ph + 1) * (self.width - pw + 1) == 1:
                return location

    def _compose(self, patches, indices, locations=None):
        indices = list(indices)
        output = self.images[indices].clone()
        ph, pw = self.patch_size
        locations = (
            [self.locations[index] for index in indices]
            if locations is None
            else locations
        )
        for row, (location, patch) in enumerate(zip(locations, patches)):
            top, left = location
            output[row, :, top:top + ph, left:left + pw] = patch
        return output

    def _original_crops(self, indices, locations=None):
        ph, pw = self.patch_size
        locations = (
            [self.locations[index] for index in indices]
            if locations is None
            else locations
        )
        return torch.stack(
            [
                self.images[index, :, top:top + ph, left:left + pw]
                for index, (top, left) in zip(indices, locations)
            ]
        )

    def _is_success_tensor(self, predictions, indices):
        labels = (
            self.target_labels[indices]
            if self.targeted
            else self.true_labels[indices]
        )
        return predictions.eq(labels) if self.targeted else predictions.ne(labels)

    def _forward(self, images, sample_ids):
        logits = self.model.forward(images)
        predictions = logits.argmax(dim=1)
        for row, sample_id in enumerate(sample_ids):
            self.query_count[sample_id] += 1
            success = bool(
                self._is_success_tensor(
                    predictions[row:row + 1], [sample_id]
                )[0].item()
            )
            if success and self.first_success_query[sample_id] is None:
                self.first_success_query[sample_id] = self.query_count[sample_id]
        return logits

    def _ce_objective(self, logits, indices):
        labels = (
            self.target_labels[indices]
            if self.targeted
            else self.true_labels[indices]
        )
        ce = F.cross_entropy(logits, labels, reduction="none")
        return -ce if self.targeted else ce

    def _margin_objective(self, logits, indices):
        rows = torch.arange(len(indices), device=self.device)
        if self.targeted:
            return (
                logits[rows, self.target_labels[indices]]
                - logits[rows, self.true_labels[indices]]
            )
        other = logits.clone()
        other[rows, self.true_labels[indices]] = -torch.inf
        return other.max(dim=1).values - logits[
            rows, self.true_labels[indices]
        ]

    def _is_better(self, index, success, loss, l2):
        if self.best_patch[index] is None:
            return True
        if not self.best_success[index] and success:
            return True
        if self.best_success[index] and success:
            return l2 < self.best_l2[index]
        if not self.best_success[index] and not success:
            return loss < self.best_loss[index]
        return False

    def _record(self, iteration, patches, logits, objectives, indices):
        predictions = logits.argmax(dim=1)
        successes = self._is_success_tensor(predictions, indices)
        originals = self._original_crops(indices)
        distances = torch.linalg.vector_norm(
            (patches - originals).flatten(1), dim=1
        )
        for row, index in enumerate(indices):
            success = bool(successes[row].item())
            loss = float((-objectives[row]).item())
            l2 = float(distances[row].item())
            if self._is_better(index, success, loss, l2):
                self.best_patch[index] = patches[row].detach().clone()
                self.best_location[index] = list(self.locations[index])
                self.best_success[index] = success
                self.best_prediction[index] = int(predictions[row].item())
                self.best_l2[index] = l2
                self.best_loss[index] = loss
            patch = self.best_patch[index].permute(1, 2, 0).cpu().numpy().copy()
            self.process[index].append(
                [
                    iteration,
                    self.best_success[index],
                    self.best_location[index].copy(),
                    patch,
                    self.best_l2[index],
                    self.best_loss[index],
                ]
            )
        return successes

    def _move_best(self, patches, indices, project_eps=False):
        ph, pw = self.patch_size
        for row, index in enumerate(indices):
            self.locations[index] = self._sample_location(
                exclude=self.locations[index]
            )
            patches[row] = self.best_patch[index]
        if project_eps:
            originals = self._original_crops(indices)
            patches = originals + (patches - originals).clamp(
                -self.eps, self.eps
            )
        return patches.clamp(0.0, 1.0)

    def _results(self):
        results = []
        ph, pw = self.patch_size
        for index in range(self.batch_size):
            image = self.images[index].clone()
            top, left = self.best_location[index]
            image[:, top:top + ph, left:left + pw] = self.best_patch[index]
            results.append(
                {
                    "adversarial": self.best_success[index],
                    "prediction": self.best_prediction[index],
                    "location": self.best_location[index].copy(),
                    "patch": self.best_patch[index]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .copy(),
                    "image": image.permute(1, 2, 0).cpu().numpy(),
                    "l2": self.best_l2[index],
                    "loss": self.best_loss[index],
                    "queries": self.query_count[index],
                    "first_success_query": self.first_success_query[index],
                    "process": self.process[index],
                }
            )
        return results


class BatchedMaskedPGD(BatchedWhiteBoxPatchAttack):
    def run(self):
        indices = list(range(self.batch_size))
        originals = self._original_crops(indices)
        patches = (
            originals
            + torch.empty_like(originals).uniform_(-self.eps, self.eps)
        ).clamp(0.0, 1.0)
        active = indices
        for iteration in tqdm(range(1, self.steps + 1)):
            if (
                self.location_update_period > 0
                and iteration > 1
                and (iteration - 1) % self.location_update_period == 0
            ):
                active_patches = self._move_best(
                    patches[active].clone(), active, project_eps=True
                )
                patches[active] = active_patches
            current = patches[active].detach().requires_grad_(True)
            logits = self._forward(self._compose(current, active), active)
            objectives = self._ce_objective(logits, active)
            successes = self._record(
                iteration, current.detach(), logits, objectives, active
            )
            if self.early_stop:
                active = [
                    index
                    for row, index in enumerate(active)
                    if not bool(successes[row].item())
                ]
                if not active:
                    break
            if iteration == self.steps:
                break
            gradient = torch.autograd.grad(objectives.sum(), current)[0]
            originals = self._original_crops(active)
            active_rows = [
                row for row, success in enumerate(successes) if not (
                    self.early_stop and bool(success.item())
                )
            ]
            candidate = current.detach()[active_rows]
            candidate += self.step_size * gradient[active_rows].sign()
            candidate = originals + (candidate - originals).clamp(
                -self.eps, self.eps
            )
            patches[active] = candidate.clamp(0.0, 1.0)
        return self._results()


class BatchedMaskedAutoPGD(BatchedWhiteBoxPatchAttack):
    def run(self):
        indices = list(range(self.batch_size))
        originals = self._original_crops(indices)
        patches = (
            originals
            + torch.empty_like(originals).uniform_(-self.eps, self.eps)
        ).clamp(0.0, 1.0)
        previous = patches.clone()
        search_best = patches.clone()
        search_best_objective = torch.full(
            (self.batch_size,), -torch.inf, device=self.device
        )
        step_sizes = torch.full(
            (self.batch_size,), self.step_size, device=self.device
        )
        histories = [[] for _ in indices]
        checkpoint = max(self.steps // 10, 1)
        active = indices
        for iteration in tqdm(range(1, self.steps + 1)):
            if (
                self.location_update_period > 0
                and iteration > 1
                and (iteration - 1) % self.location_update_period == 0
            ):
                moved = self._move_best(
                    patches[active].clone(), active, project_eps=True
                )
                patches[active] = moved
                previous[active] = moved
                search_best[active] = moved
                search_best_objective[active] = -torch.inf
                step_sizes[active] = self.step_size
                for index in active:
                    histories[index] = []
            current = patches[active].detach().requires_grad_(True)
            logits = self._forward(self._compose(current, active), active)
            objectives = self._ce_objective(logits, active)
            successes = self._record(
                iteration, current.detach(), logits, objectives, active
            )
            for row, index in enumerate(active):
                value = float(objectives[row].item())
                histories[index].append(value)
                if value > float(search_best_objective[index].item()):
                    search_best_objective[index] = objectives[row].detach()
                    search_best[index] = current[row].detach()
            kept_rows = [
                row for row, success in enumerate(successes) if not (
                    self.early_stop and bool(success.item())
                )
            ]
            active = [active[row] for row in kept_rows]
            if not active or iteration == self.steps:
                break
            gradient = torch.autograd.grad(objectives.sum(), current)[0]
            cur = current.detach()[kept_rows]
            grad = gradient[kept_rows]
            originals = self._original_crops(active)
            projected = cur + step_sizes[active, None, None, None] * grad.sign()
            projected = originals + (projected - originals).clamp(
                -self.eps, self.eps
            )
            momentum = 0.75 if iteration > 1 else 1.0
            candidate = cur + momentum * (projected - cur)
            candidate += (1.0 - momentum) * (cur - previous[active])
            candidate = originals + (candidate - originals).clamp(
                -self.eps, self.eps
            )
            previous[active] = cur
            patches[active] = candidate.clamp(0.0, 1.0)
            if iteration % checkpoint == 0:
                for index in active:
                    history = histories[index]
                    improvements = sum(
                        current_value > prior
                        for prior, current_value in zip(history, history[1:])
                    )
                    if improvements <= len(history) // 2:
                        step_sizes[index] *= 0.5
                        patches[index] = search_best[index]
                        previous[index] = search_best[index]
                    histories[index] = []
        return self._results()


class BatchedLaVAN(BatchedWhiteBoxPatchAttack):
    def _objective(self, logits, indices):
        if self.targeted:
            source = F.cross_entropy(
                logits, self.true_labels[indices], reduction="none"
            )
            target = F.cross_entropy(
                logits, self.target_labels[indices], reduction="none"
            )
            return target - source
        return self._margin_objective(logits, indices)

    def run(self):
        patches = torch.rand(
            self.batch_size,
            3,
            self.patch_size[0],
            self.patch_size[1],
            device=self.device,
        )
        active = list(range(self.batch_size))
        current = patches[active].detach().requires_grad_(True)
        logits = self._forward(self._compose(current, active), active)
        for iteration in tqdm(range(1, self.steps + 1)):
            if (
                self.location_update_period > 0
                and iteration > 1
                and (iteration - 1) % self.location_update_period == 0
            ):
                patches[active] = self._move_best(
                    patches[active].clone(), active
                )
                current = patches[active].detach().requires_grad_(True)
                logits = self._forward(
                    self._compose(current, active), active
                )
            objectives = self._objective(logits, active)
            gradient = torch.autograd.grad(objectives.sum(), current)[0]
            updated = (
                current + self.step_size * gradient
            ).clamp(0.0, 1.0).detach().requires_grad_(True)
            logits = self._forward(
                self._compose(updated, active), active
            )
            objectives = self._objective(logits, active)
            successes = self._record(
                iteration, updated.detach(), logits, objectives, active
            )
            patches[active] = updated.detach()
            if self.early_stop:
                kept_rows = [
                    row
                    for row, success in enumerate(successes)
                    if not bool(success.item())
                ]
                active = [active[row] for row in kept_rows]
                if not active:
                    break
                current = patches[active].detach().requires_grad_(True)
                logits = self._forward(
                    self._compose(current, active), active
                )
            else:
                current = updated
        return self._results()


class BatchedLOAP(BatchedWhiteBoxPatchAttack):
    _DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def __init__(
        self,
        *args,
        lo_mode="full",
        stride=2,
        attempts=1,
        exclude_box=None,
        **kwargs,
    ):
        self.lo_mode = lo_mode
        self.stride = int(stride)
        self.attempts = int(attempts)
        self.exclude_box = (
            None if exclude_box is None else tuple(exclude_box)
        )
        self._allowed_locations_cache = None
        super().__init__(*args, **kwargs)
        if self.lo_mode not in {"full", "random"}:
            raise ValueError("lo_mode must be 'full' or 'random'")
        if not self._allowed_locations_cache:
            raise ValueError("exclude_box leaves no valid patch location")

    def _allowed(self, location):
        top, left = location
        ph, pw = self.patch_size
        if top < 0 or left < 0 or top + ph > self.height or left + pw > self.width:
            return False
        if self.exclude_box is None:
            return True
        ex_top, ex_left, ex_h, ex_w = self.exclude_box
        return (
            top + ph <= ex_top
            or top >= ex_top + ex_h
            or left + pw <= ex_left
            or left >= ex_left + ex_w
        )

    def _sample_location(self, exclude=None):
        if self._allowed_locations_cache is None:
            self._allowed_locations_cache = [
                (top, left)
                for top in range(self.height - self.patch_size[0] + 1)
                for left in range(self.width - self.patch_size[1] + 1)
                if self._allowed((top, left))
            ]
        locations = self._allowed_locations_cache
        if exclude is not None and len(locations) > 1:
            locations = [
                location
                for location in locations
                if location != tuple(exclude)
            ]
        return random.choice(locations)

    def _optimize_locations(self, patches, indices):
        current_logits = self._forward(
            self._compose(patches, indices), indices
        )
        best_values = self._ce_objective(current_logits, indices).detach()
        candidates, candidate_ids, candidate_locations = [], [], []
        for row, index in enumerate(indices):
            directions = (
                self._DIRECTIONS
                if self.lo_mode == "full"
                else (random.choice(self._DIRECTIONS),)
            )
            top, left = self.locations[index]
            for row_delta, left_delta in directions:
                location = (
                    top + row_delta * self.stride,
                    left + left_delta * self.stride,
                )
                if self._allowed(location):
                    candidates.append(
                        self._compose(
                            patches[row:row + 1],
                            [index],
                            [location],
                        )[0]
                    )
                    candidate_ids.append(index)
                    candidate_locations.append(location)
        if not candidates:
            return
        logits = self._forward(torch.stack(candidates), candidate_ids)
        values = self._ce_objective(logits, candidate_ids)
        row_by_id = {index: row for row, index in enumerate(indices)}
        for row, (index, location) in enumerate(
            zip(candidate_ids, candidate_locations)
        ):
            index_row = row_by_id[index]
            if values[row] > best_values[index_row]:
                best_values[index_row] = values[row]
                self.locations[index] = location

    def run(self):
        active = list(range(self.batch_size))
        global_iteration = 0
        for _ in range(self.attempts):
            for index in active:
                self.locations[index] = self._sample_location()
            patches = torch.rand(
                self.batch_size,
                3,
                self.patch_size[0],
                self.patch_size[1],
                device=self.device,
            )
            for _ in tqdm(range(self.steps)):
                global_iteration += 1
                current = patches[active].detach().requires_grad_(True)
                logits = self._forward(self._compose(current, active), active)
                objectives = self._ce_objective(logits, active)
                gradient = torch.autograd.grad(objectives.sum(), current)[0]
                updated = (
                    current + self.step_size * gradient.sign()
                ).clamp(0.0, 1.0).detach()
                self._optimize_locations(updated, active)
                logits = self._forward(
                    self._compose(updated, active), active
                )
                objectives = self._ce_objective(logits, active)
                successes = self._record(
                    global_iteration, updated, logits, objectives, active
                )
                patches[active] = updated
                if self.early_stop:
                    active = [
                        index
                        for row, index in enumerate(active)
                        if not bool(successes[row].item())
                    ]
                    if not active:
                        break
            if not active:
                break
        return self._results()
