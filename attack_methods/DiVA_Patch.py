import numpy as np
from tqdm import tqdm
from scipy.special import softmax

from attack_methods.base import Attacker
from attack_methods.utils import l2_compute
from attack_methods.CamoPatch import mutate

from base import MAP_Elites
from base.individual import Individual, render

class DiVA_Patch(Attacker):
    def __init__(self, img_cls, loss_function, delta_max=20, delta_min=0.1, K=100,
                 patch_size=(40, 40), grid_size=(40, 40), max_query=10000, setting='realistic'):
        super().__init__(max_query)

        self.img_cls = img_cls
        self.loss_function = loss_function
        self.loss_function.bind_base_image(img_cls)
        self.h, self.w = img_cls.shape[0], img_cls.shape[1]

        self.patch_size = patch_size
        self.grid_size_h, self.grid_size_w = grid_size[0], grid_size[1]

        self.map_elites = MAP_Elites(img_h=self.h, img_w=self.w, cell_h=self.grid_size_h, cell_w=self.grid_size_w)

        self.pbar = tqdm(total=self.max_query)
        self.delta_max, self.delta_min = delta_max, delta_min
        self.K = K

        self.setting = setting

    def evaluate(self, idv):
        s = idv.s
        x, y = idv.location

        orig_patch = self.img_cls[x: x + s[0], y: y + s[1], :]

        adversarial, loss = self.loss_function.evaluate_patch(
            idv.patch, idv.location
        )
        self.count_query(adversarial)
        self.pbar.update(1)
        l2_score = l2_compute(adv_patch=idv.patch, orig_patch=orig_patch, integer=(self.setting == 'realistic'))

        idv.success_attack = adversarial
        idv.loss, idv.l2 = loss, l2_score

    def run(self):
        # Step 1: Initialize Archive
        self.initialize_archive()
        best_idv = self.get_best_quality_solution()
        self.process.append([
            self.n_query,
            best_idv.success_attack, best_idv.location, best_idv.patch,
            best_idv.l2, best_idv.loss]
        )

        first_print = False
        while self.n_query < self.max_query:
            # Step 2: Select Niche
            idx = self.select_niche()

            # Step 3: Modify Niche
            self.modify_niche(idx)

            best_idv = self.get_best_quality_solution()
            self.process.append([
                self.n_query,
                best_idv.success_attack, best_idv.location, best_idv.patch,
                best_idv.l2, best_idv.loss]
            )

            if best_idv.success_attack and not first_print:
                print(f'Successfully attack at {self.n_query} #evals!')
                first_print = True
        self.pbar.close()

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
            idv = Individual(patch_size=self.patch_size, setting=self.setting)

            idv.rand(img_h=self.h - idv.s[0], img_w=self.w - idv.s[1], lx=lx, ly=ly, ux=ux, uy=uy)

            self.evaluate(idv)
            self.map_elites.assign(idv)

    # Step 2: Select-Niche
    def select_niche(self):
        list_activated_cell, list_loss = self.get_modified_loss()
        delta = compute_delta(self.max_query - self.n_query, self.max_query,
                              delta_max=self.delta_max, delta_min=self.delta_min)
        list_probs = softmax(-list_loss / delta)

        idx = np.random.choice(list_activated_cell, p=list_probs)

        return idx

    def get_modified_loss(self):
        list_activated_cell, list_success_loss = [], []
        for idx in self.map_elites.activated_cells:
            list_activated_cell.append(idx)
            if self.map_elites.grid_search[idx].success_attack:
                list_success_loss.append(self.map_elites.grid_search[idx].loss)

        max_success_loss = None
        if len(list_success_loss) != 0:
            max_success_loss = max(list_success_loss)

        list_loss = []
        for idx in list_activated_cell:
            if self.map_elites.grid_search[idx].success_attack:
                list_loss.append(max_success_loss)
            else:
                list_loss.append(self.map_elites.grid_search[idx].loss)

        list_loss = np.array(list_loss)
        return list_activated_cell, list_loss

    # Step 3: Modify-Niche
    def modify_niche(self, idx):
        idx_best = self.map_elites.idx_best_search

        if idx == idx_best or self.map_elites.grid_search[idx].success_attack:
            # Exploitation
            self.modify_best_patch(idx)
        else:
            # Exploration
            self.modify_unsuccessful_patch(idx)

    def modify_best_patch(self, idx):
        """
            Algorithm 3 (Line 3 - 9)
        """
        idv = self.map_elites.grid_search[idx]

        if not idv.success_attack:
            o = Individual(patch_size=self.patch_size, setting=self.setting)

            o.location = idv.location.copy()
            o.s = idv.s.copy()
            o.patch_geno = mutate(idv.patch_geno, setting=self.setting)
            o.patch = render(o.patch_geno, o.s[0], o.s[1], setting=self.setting)

            self.evaluate(o)
            self.map_elites.assign(o)
        else:
            best_patch_geno = idv.patch_geno.copy()
            best_l2 = idv.l2

            for _ in range(self.K):
                o = Individual(patch_size=self.patch_size, setting=self.setting)
                o.location = idv.location.copy()
                o.s = idv.s.copy()
                o.patch_geno = mutate(best_patch_geno, setting=self.setting)
                o.patch = render(o.patch_geno, o.s[0], o.s[1], setting=self.setting)
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

        o = Individual(patch_size=self.patch_size, setting=self.setting)
        if np.random.random() <= 0.5:
            # Mutate location
            o.location = update_location(idv.location.copy(), self.h, self.w, idv.s[0], idv.s[1])
            o.patch_geno = idv.patch_geno.copy()
            o.patch = idv.patch.copy()
        else:
            # Mutate patch
            o.location = idv.location.copy()
            o.s = idv.s.copy()
            o.patch_geno = mutate(idv.patch_geno, setting=self.setting)
            o.patch = render(o.patch_geno, o.s[0], o.s[1], setting=self.setting)
        self.evaluate(o)
        self.map_elites.assign(o)

    def get_best_quality_solution(self):
        idx = self.map_elites.idx_best_final
        return self.map_elites.grid_final[idx]

    def return_map_elites(self):
        data = []
        for idx, idv in self.map_elites.grid_final.items():
            if idv is not None:
                data.append([idx, idv.success_attack, idv.location, idv.patch, idv.l2, idv.loss])
            else:
                data.append([idx, None, None, None, None, None, None])
        return data


def update_location(loc_new, h, w, s_h, s_w):
    loc_new[0] = np.random.randint(low=0, high=h - s_h)
    loc_new[1] = np.random.randint(low=0, high=w - s_w)
    return loc_new

def compute_delta(remaining_evals, total_evals, delta_min=0.1, delta_max=10.0):
    """
    Computes temperature delta scaled to remaining evaluations.

    Parameters:
    - remaining_evals: number of evaluations left
    - total_evals: total number of evaluations planned
    - delta_min: minimum temperature (at the end → exploit)
    - delta_max: maximum temperature (at the start → explore)

    Returns:
    - current delta
    """
    progress_ratio = remaining_evals / (total_evals + 1e-8)
    T = delta_min + (delta_max - delta_min) * (progress_ratio ** 20)
    return T
