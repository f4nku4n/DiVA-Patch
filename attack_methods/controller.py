from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class AttackJob:
    attacker: object
    context: object = None


class AttackController:
    """Keep a window of ask/tell attackers supplied with batched evaluations."""

    def __init__(self, evaluator, batch_size=1, on_complete=None, show_progress=True):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.evaluator = evaluator
        self.batch_size = int(batch_size)
        self.on_complete = on_complete
        self.show_progress = show_progress

    def _complete(self, job):
        if self.on_complete is not None:
            self.on_complete(job)

    def run(self, jobs):
        jobs = iter(jobs)
        active = []
        source_exhausted = False

        def fill_active():
            nonlocal source_exhausted
            while not source_exhausted and len(active) < self.batch_size:
                try:
                    active.append(next(jobs))
                except StopIteration:
                    source_exhausted = True

        progress = tqdm(unit="query", desc="DiVA-Patch", disable=not self.show_progress)
        try:
            fill_active()
            while active:
                # An attacker normally becomes done in tell(). This loop also
                # handles an initially-complete generic attacker without wasting
                # a batch slot.
                while True:
                    completed = []
                    for job in active:
                        if job.attacker.ask() is None:
                            completed.append(job)
                    if not completed:
                        break
                    for job in completed:
                        active.remove(job)
                        self._complete(job)
                    fill_active()
                    if not active:
                        break

                if not active:
                    break

                queries = [job.attacker.ask() for job in active]
                results = self.evaluator.evaluate(queries)
                if len(results) != len(active):
                    raise RuntimeError(
                        f"Evaluator returned {len(results)} results for {len(active)} queries"
                    )

                completed = []
                for job, result in zip(active, results):
                    job.attacker.tell(result)
                    if job.attacker.done:
                        completed.append(job)
                progress.update(len(queries))

                for job in completed:
                    active.remove(job)
                    self._complete(job)
                fill_active()
        finally:
            progress.close()
