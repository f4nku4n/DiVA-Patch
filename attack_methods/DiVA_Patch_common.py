import numpy as np
from base.individual_common import Individual
import cv2

from .utils import l2_compute
from .DiVA_Patch import DiVA_Patch, update_location


class DiVA_Patch_common(DiVA_Patch):
    def __init__(self, img_cls, loss_function, delta_max=10, delta_min=0.1, K=100,
                 patch_size=[40, 40], grid_size=[40, 40], max_query=10000):
        super().__init__(img_cls, loss_function, delta_max, delta_min, K,
                         patch_size, grid_size, max_query)

    def evaluate(self, idv):
        self.n_query += 1
        self.pbar.update(1)

        s = idv.s
        x, y = idv.location

        orig_patch = self.img_cls[x: x + s[0], y: y + s[1], :]

        adversarial, loss = self.loss_function.evaluate_patch(
            idv.patch, idv.location
        )
        l2_score = l2_compute(adv_patch=idv.patch, orig_patch=orig_patch)  # Difference

        idv.success_attack = adversarial
        idv.loss, idv.l2 = loss, l2_score

    # Step 1: Initialize Archive
    def initialize_archive(self):
        for idx in self.map_elites.grid_search:
            i, j = int(idx.split('-')[0]), int(idx.split('-')[1])
            lx, ux = i * self.map_elites.cell_h, (i + 1) * self.map_elites.cell_h
            ly, uy = j * self.map_elites.cell_w, (j + 1) * self.map_elites.cell_w

            if (lx >= self.h - self.patch_size[0]) or (ly >= self.w - self.patch_size[1]):
                continue

            ux = min(ux + 1, self.h - self.patch_size[0])
            uy = min(uy + 1, self.w - self.patch_size[1])
            idv = Individual(patch_size=self.patch_size)

            idv.rand(img_h=self.h - idv.s[0], img_w=self.w - idv.s[1], lx=lx, ly=ly, ux=ux, uy=uy)

            self.evaluate(idv)
            self.map_elites.assign(idv)

    def modify_best_patch(self, idx):
        idv = self.map_elites.grid_search[idx]

        if not idv.success_attack:
            o = Individual(patch_size=self.patch_size)

            o.location = idv.location.copy()
            o.s = idv.s.copy()
            o.patch_geno = mutate_patch(idv.patch_geno)  # Difference
            o.patch = render(o.patch_geno, o.s[0], o.s[1])

            self.evaluate(o)
            self.map_elites.assign(o)
        else:
            best_patch_geno = idv.patch_geno.copy()
            best_l2 = idv.l2

            for _ in range(self.K):
                o = Individual(patch_size=self.patch_size)
                o.location = idv.location.copy()
                o.s = idv.s.copy()
                o.patch_geno = mutate_patch(best_patch_geno)  # Difference
                o.patch = render(o.patch_geno, o.s[0], o.s[1])  # Difference
                self.evaluate(o)
                self.map_elites.assign(o)

                if o.success_attack and o.l2 < best_l2:
                    best_patch_geno = o.patch_geno.copy()
                    best_l2 = o.l2

    def modify_unsuccessful_patch(self, idx):
        """
        Algorithm 3 (Line 11 - 18)
        """
        idv = self.map_elites.grid_search[idx]

        o = Individual(patch_size=self.patch_size)
        if np.random.random() <= 0.5:
            # Mutate location
            o.location = update_location(idv.location.copy(), self.h, self.w, idv.s[0], idv.s[1])
            o.patch_geno = idv.patch_geno.copy()
            o.patch = idv.patch.copy()
        else:
            # Mutate patch
            o.location = idv.location.copy()
            o.s = idv.s.copy()
            o.patch_geno = mutate_patch(idv.patch_geno)
            o.patch = render(o.patch_geno, o.s[0], o.s[1])
        self.evaluate(o)
        self.map_elites.assign(o)


def mutate_patch(soln, mut=0.3):
    new_specie = soln.copy()

    genes = soln.shape[0]
    length = soln.shape[1]
    y = np.random.randint(0, genes)
    change = np.random.randint(0, length + 1)

    selection = np.random.choice(length, size=change, replace=False)

    if np.random.rand() < mut:
        new_specie[y, selection] = np.random.rand(len(selection))
    else:
        new_specie[y, selection] += (np.random.rand(len(selection)) - 0.5) / 3
        new_specie[y, selection] = np.clip(new_specie[y, selection], 0, 1)

    return new_specie


def render(x, h, w):
    phenotype = np.ones((h, w, 3))  # load a white patch w*w
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        cv2.circle(
            overlay,
            center=(int(row[1] * h), int(row[0] * w)),
            radius=int(row[2] * radius_avg),
            color=(int(row[3] * 255), int(row[4] * 255), int(row[5] * 255)),
            thickness=-1,
        )
        alpha = row[6]
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)

    return phenotype / 255.
