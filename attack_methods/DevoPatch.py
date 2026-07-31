import numpy as np
from tqdm import tqdm

from attack_methods.base import Attacker


class DevoPatch(Attacker):
    """Decision-based differential-evolution adversarial patch attack."""

    def __init__(
        self,
        img_cls,
        target_image,
        loss_function,
        pop_size=10,
        init_rate=0.35,
        mutation_rate=1,
        fitness_norm=0,
        max_query=10000,
    ):
        super().__init__(max_query)
        if pop_size <= 0:
            raise ValueError("pop_size must be positive")
        if not 0 < init_rate <= 1:
            raise ValueError("init_rate must be in (0, 1]")
        if mutation_rate < 0 or int(mutation_rate) != mutation_rate:
            raise ValueError("mutation_rate must be a non-negative integer")
        if fitness_norm not in (0, 1, 2):
            raise ValueError("fitness_norm must be 0, 1, or 2")
        if img_cls.shape != target_image.shape:
            raise ValueError("source and target images must have the same shape")

        self.img_cls = np.asarray(img_cls)
        self.target_image = np.asarray(target_image)
        self.loss_function = loss_function
        self.pop_size = int(pop_size)
        self.init_rate = float(init_rate)
        self.mutation_rate = int(mutation_rate)
        self.fitness_norm = fitness_norm
        self.h, self.w = self.img_cls.shape[:2]
        self.value_scale = (
            255.0
            if max(
                float(np.max(self.img_cls)),
                float(np.max(self.target_image)),
            ) > 1.5
            else 1.0
        )

        self.loss_function.bind_base_image(self.img_cls)
        self.loss_function.bind_target_image(self.target_image)

        self.population = []
        self.fitnesses = []
        self.best_rectangle = np.array([0, 0, self.h, self.w], dtype=np.int64)
        self.best_success = False
        self.best_fitness = float("inf")
        self.best_l2 = self._l2(self.best_rectangle)

    def _patch(self, rectangle):
        top, left, bottom, right = rectangle
        return self.target_image[top:bottom, left:right].copy()

    def _difference(self, rectangle):
        top, left, bottom, right = rectangle
        return (
            self.target_image[top:bottom, left:right].astype(np.float64)
            - self.img_cls[top:bottom, left:right].astype(np.float64)
        ) / self.value_scale

    def _fitness(self, rectangle):
        difference = self._difference(rectangle).ravel()
        if self.fitness_norm == 0:
            return float(np.count_nonzero(difference))
        return float(np.linalg.norm(difference, ord=self.fitness_norm))

    def _l2(self, rectangle):
        return float(np.linalg.norm(self._difference(rectangle).ravel(), ord=2))

    def _record(self):
        rectangle = self.best_rectangle.copy()
        self.process.append([
            self.n_query,
            self.best_success,
            rectangle[:2].tolist(),
            self._patch(rectangle),
            self.best_l2,
            self.best_fitness,
        ])

    def _evaluate(self, rectangle):
        if self.n_query >= self.max_query:
            return None
        rectangle = self._bound_handle(np.asarray(rectangle, dtype=np.int64))
        success = self.loss_function.evaluate_rectangle(rectangle)
        self.count_query(success)
        fitness = self._fitness(rectangle)
        l2 = self._l2(rectangle)

        if (
            success
            and (
                not self.best_success
                or fitness < self.best_fitness
            )
        ):
            self.best_rectangle = rectangle.copy()
            self.best_success = True
            self.best_fitness = fitness
            self.best_l2 = l2
        elif not self.best_success:
            self.best_rectangle = rectangle.copy()
            self.best_fitness = float("inf")
            self.best_l2 = l2
        self._record()
        return success, fitness, rectangle

    def _bound_handle(self, rectangle):
        top, left, bottom, right = rectangle.tolist()
        top, bottom = sorted((top, bottom))
        left, right = sorted((left, right))
        top = int(np.clip(top, 0, self.h - 1))
        left = int(np.clip(left, 0, self.w - 1))
        bottom = int(np.clip(bottom, top + 1, self.h))
        right = int(np.clip(right, left + 1, self.w))
        return np.array([top, left, bottom, right], dtype=np.int64)

    def _initial_rectangle(self, retries):
        height_margin = max(1, int(self.h * self.init_rate))
        width_margin = max(1, int(self.w * self.init_rate))
        if retries > 10:
            height_margin = width_margin = 1
        return np.array([
            np.random.randint(0, height_margin),
            np.random.randint(0, width_margin),
            np.random.randint(self.h - height_margin + 1, self.h + 1),
            np.random.randint(self.w - width_margin + 1, self.w + 1),
        ], dtype=np.int64)

    def _initialize_population(self, progress):
        for _ in range(self.pop_size):
            retries = 0
            while self.n_query < self.max_query:
                result = self._evaluate(self._initial_rectangle(retries))
                progress.update(1)
                if result[0]:
                    self.population.append(result[2])
                    self.fitnesses.append(result[1])
                    break
                retries += 1
            if self.n_query >= self.max_query:
                break

    def _offspring(self):
        fitnesses = np.asarray(self.fitnesses)
        best_index = int(np.argmin(fitnesses))
        choices = [i for i in range(len(self.population)) if i != best_index]
        j, q = np.random.choice(choices, size=2, replace=False)
        offspring = (
            self.population[best_index]
            + self.mutation_rate
            * (self.population[int(j)] - self.population[int(q)])
        )
        noise = np.random.randint(
            -self.mutation_rate,
            self.mutation_rate + 1,
            size=4,
        )
        return self._bound_handle(offspring + noise)

    def run(self):
        with tqdm(total=self.max_query, unit="query") as progress:
            self._initialize_population(progress)
            while self.n_query < self.max_query and len(self.population) >= 3:
                result = self._evaluate(self._offspring())
                progress.update(1)
                success, fitness, rectangle = result
                if success:
                    worst_index = int(np.argmax(self.fitnesses))
                    if fitness < self.fitnesses[worst_index]:
                        self.population[worst_index] = rectangle
                        self.fitnesses[worst_index] = fitness
        return self.get_best()

    def get_best(self):
        rectangle = self.best_rectangle.copy()
        top, left, bottom, right = rectangle
        return {
            "success": self.best_success,
            "rectangle": rectangle,
            "location": [int(top), int(left)],
            "patch": self._patch(rectangle),
            "patch_area": int((bottom - top) * (right - left)),
            "patch_area_ratio": float(
                (bottom - top) * (right - left) / (self.h * self.w)
            ),
            "fitness": self.best_fitness,
            "l2": self.best_l2,
            "queries": self.n_query,
            "first_success_query": self.first_success_query,
        }

    def build_adversarial(self):
        image = self.img_cls.copy()
        top, left, bottom, right = self.best_rectangle
        image[top:bottom, left:right] = self.target_image[
            top:bottom, left:right
        ]
        return image
