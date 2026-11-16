from dataclasses import dataclass


@dataclass(frozen=True)
class WhiteBoxQuery:
    image: object
    true_label: int
    target_label: int


class WhiteBoxAttacker:
    """One-image ask/tell facade for a batched white-box optimizer."""

    def __init__(self, image, true_label, target_label):
        self.query = WhiteBoxQuery(image, int(true_label), int(target_label))
        self.result = None
        self._pending = False

    @property
    def done(self):
        return self.result is not None

    def ask(self):
        if self.done:
            return None
        self._pending = True
        return self.query

    def tell(self, result):
        if not self._pending or self.done:
            raise RuntimeError("tell() requires a pending white-box query")
        self._pending = False
        self.result = result


class WhiteBoxEvaluator:
    """Run N independent image attacks through one batched implementation."""

    def __init__(self, attack_class, model, attack_kwargs):
        self.attack_class = attack_class
        self.model = model
        self.attack_kwargs = dict(attack_kwargs)

    def evaluate(self, queries):
        if not queries:
            return []
        import numpy as np
        import torch

        try:
            attack = self.attack_class(
                images=np.stack([query.image for query in queries]),
                model=self.model,
                true_labels=[query.true_label for query in queries],
                target_labels=[query.target_label for query in queries],
                **self.attack_kwargs,
            )
            results = attack.run()
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                f"CUDA out of memory during white-box evaluator batch of "
                f"{len(queries)} images; reduce --batch_size and rerun."
            ) from error
        if len(results) != len(queries):
            raise RuntimeError(
                f"White-box evaluator returned {len(results)} results for "
                f"{len(queries)} attackers"
            )
        return results
