from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class PatchQuery:
    base_image: torch.Tensor
    patch: np.ndarray
    location: tuple
    targeted: bool
    true_label: int
    target_label: int = None
    unnormalized: bool = False
    clip_bounds: tuple = None


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    loss: float


def _candidate_image(query):
    image = query.base_image.clone()
    patch = query.patch * 255.0 if query.unnormalized else query.patch
    patch_tensor = torch.from_numpy(np.asarray(patch)).permute(2, 0, 1).to(
        device=image.device,
        dtype=image.dtype,
    )
    x, y = (int(value) for value in query.location)
    patch_h, patch_w = patch_tensor.shape[-2:]
    image[:, :, x:x + patch_h, y:y + patch_w] = patch_tensor.unsqueeze(0)

    if query.clip_bounds is not None:
        bounds = tuple(
            bound * 255.0 if query.unnormalized else bound
            for bound in query.clip_bounds
        )
        image.clamp_(min=bounds[0], max=bounds[1])
    return image


class BatchEvaluator:
    """Evaluate independent attack queries with one victim-model batch."""

    def __init__(self, model):
        self.model = model

    def _predict_many(self, images):
        if hasattr(self.model, "predict_many"):
            return self.model.predict_many(images)

        images = list(images)
        shapes = {tuple(image.shape[-2:]) for image in images}
        if len(shapes) != 1:
            raise ValueError(
                "The victim model must implement predict_many() to batch images "
                "with different spatial sizes"
            )
        return self.model.predict(torch.cat(images, dim=0))

    @staticmethod
    def _raise_oom(error, batch_size):
        raise RuntimeError(
            f"BatchEvaluator ran out of CUDA memory for batch_size={batch_size}. "
            "Reduce --batch_size and run again."
        ) from error

    def predict_labels(self, images):
        if not images:
            return []
        try:
            predictions = self._predict_many(images)
            return torch.argmax(predictions, dim=1).tolist()
        except torch.OutOfMemoryError as error:
            self._raise_oom(error, len(images))

    def evaluate(self, queries):
        if not queries:
            return []
        try:
            images = [_candidate_image(query) for query in queries]
            logits = self._predict_many(images)
            if logits.ndim != 2 or logits.shape[0] != len(queries):
                raise ValueError(
                    "Victim model returned logits with shape "
                    f"{tuple(logits.shape)} for {len(queries)} queries"
                )

            device = logits.device
            true_labels = torch.tensor(
                [query.true_label for query in queries], dtype=torch.long, device=device
            )
            targeted = torch.tensor(
                [query.targeted for query in queries], dtype=torch.bool, device=device
            )
            target_labels = torch.tensor(
                [
                    query.target_label if query.target_label is not None else query.true_label
                    for query in queries
                ],
                dtype=torch.long,
                device=device,
            )

            predictions = torch.argmax(logits, dim=1)

            true_logits = logits.gather(1, true_labels[:, None]).squeeze(1)
            other_logits = logits.clone()
            other_logits.scatter_(1, true_labels[:, None], -torch.inf)
            log_epsilon = logits.new_tensor(math.log(1e-30))
            untargeted_loss = torch.logaddexp(true_logits, log_epsilon) - torch.logaddexp(
                other_logits.max(dim=1).values, log_epsilon
            )

            target_logits = logits.gather(1, target_labels[:, None]).squeeze(1)
            targeted_loss = torch.logsumexp(logits, dim=1) - target_logits

            success = torch.where(
                targeted,
                predictions == target_labels,
                predictions != true_labels,
            )
            losses = torch.where(targeted, targeted_loss, untargeted_loss)

            # Transfer all scalar results in one synchronization.
            packed = torch.stack((success.to(logits.dtype), losses), dim=1).tolist()
            return [
                EvaluationResult(success=bool(item[0]), loss=float(item[1]))
                for item in packed
            ]
        except torch.OutOfMemoryError as error:
            self._raise_oom(error, len(queries))
