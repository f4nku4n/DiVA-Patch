import cv2
import math
import numpy as np
from tqdm import tqdm
from .base import Attacker
from .utils import l2_compute, sh_selection


class CamoPatch(Attacker):
    def __init__(self, params, loss_function, max_query=10000):
        super().__init__(max_query)
        self.loss_function = loss_function
        self.params = params

    def completion_procedure(self, adversarial, x_adv, queries, loc, patch):
        data = {
            "adversarial": adversarial,
            "queries": queries,
            "loc": loc,
            "patch": patch,
            "final_prediction": self.loss_function.get_label(x_adv),
            "process": self.process
        }

        np.save(self.params["save_directory"], data, allow_pickle=True)

    def run(self):
        x = self.params["x"]

        c, h, w = self.params["c"], self.params["h"], self.params["w"]
        eps = self.params["eps"]
        s = int(math.ceil(eps ** .5))

        patch_geno = np.random.randint(0, 256, size=(self.params["N"], 7))
        patch = render(patch_geno, s)
        loc = [np.random.randint(h - s), np.random.randint(w - s)]

        update_loc_period = self.params["update_loc_period"]

        x_adv = x.copy()
        x_adv[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch
        x_adv = np.clip(x_adv, 0., 255.)
        adversarial, loss = self.loss_function(x_adv)

        l2_curr = l2_compute(
            adv_patch=patch,
            orig_patch=x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy(),
            integer=True
        )
        self.process.append([adversarial, loc, patch_geno, l2_curr, loss])

        patch_counter = 0

        for it in tqdm(range(1, self.max_query)):
            patch_counter += 1
            if patch_counter < update_loc_period:
                patch_new_geno = mutate(patch_geno, self.params["mut"])
                patch_new = render(patch_new_geno, s)
                x_adv_new = x.copy()
                x_adv_new[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch_new
                x_adv_new = np.clip(x_adv_new, 0, 255)

                # evaluate new solutions
                adversarial_new, loss_new = self.loss_function(x_adv_new)

                orig_patch = x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy()

                l2_new = l2_compute(
                    adv_patch=patch_new,
                    orig_patch=orig_patch,
                    integer=True
                )

                if adversarial == True and adversarial_new == True:
                    if l2_new < l2_curr:
                        loss = loss_new
                        adversarial = adversarial_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new
                else:
                    if loss_new < loss:  # minimization
                        loss = loss_new
                        adversarial = adversarial_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new

            else:
                patch_counter = 0

                # location update
                sh_i = int(max(sh_selection(self.max_query, it) * h, 0))
                sw_i = int(max(sh_selection(self.max_query, it) * w, 0))
                loc_new = loc.copy()
                loc_new = update_location(loc_new, sh_i, sw_i, h, w, s)
                x_adv_new = x.copy()
                x_adv_new[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :] = patch
                x_adv_new = np.clip(x_adv_new, 0., 255.)

                # evaluate new solution
                adversarial_new, loss_new = self.loss_function(x_adv_new)

                orig_patch_new = x[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :].copy()
                l2_new = l2_compute(
                    adv_patch=patch,
                    orig_patch=orig_patch_new,
                    integer=True
                )

                if adversarial == True and adversarial_new == True:
                    if l2_new < l2_curr:
                        loss = loss_new
                        adversarial = adversarial_new
                        loc = loc_new

                        x_adv = x_adv_new
                        l2_curr = l2_new
                else:
                    diff = loss_new - loss
                    curr_temp = self.params["temp"] / (it + 1)
                    metropolis = math.exp(-diff / curr_temp)

                    if loss_new < loss or np.random.rand() < metropolis:  # minimization # first check
                        loss = loss_new
                        adversarial = adversarial_new
                        loc = loc_new
                        x_adv = x_adv_new
                        l2_curr = l2_new
            self.process.append([adversarial, loc, patch_geno, l2_curr, loss])

        self.completion_procedure(adversarial, x_adv, self.max_query, loc, patch)
        return


def update_location(loc_new, h_i, w_i, h, w, s):
    loc_new[0] += np.random.randint(low=-h_i, high=h_i + 1)
    loc_new[0] = np.clip(loc_new[0], 0, h - s)
    loc_new[1] += np.random.randint(low=-w_i, high=w_i + 1)
    loc_new[1] = np.clip(loc_new[1], 0, w - s)
    return loc_new


def render(x, w):
    phenotype = np.ones((w, w, 3), dtype=np.uint8) * 255  # load a white patch w * w
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        cv2.circle(
            overlay,
            center=(int(row[1] * w / 255), int(row[0] * w / 255)),
            radius=int(row[2] / 255 * 2 * radius_avg),
            color=(int(row[3]), int(row[4]), int(row[5])),
            thickness=-1,
        )
        alpha = row[6] / 255.
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)

    return phenotype


def mutate(soln, mut):
    """Mutates specie for evolution.

    Args:
        specie (species.Specie): Specie to mutate.

    Returns:
        New Specie class, that has been mutated.
        :param soln:
    """
    new_specie = soln.copy()

    # Randomization for Evolution
    genes = soln.shape[0]
    length = soln.shape[1]
    y = np.random.randint(0, genes)
    change = np.random.randint(0, length + 1)

    if change >= length + 1:
        change -= 1
        i, j = y, np.random.randint(0, genes)
        i, j, s = (i, j, -1) if i < j else (j, i, 1)
        new_specie[i: j + 1] = np.roll(new_specie[i: j + 1], shift=s, axis=0)
        y = j

    selection = np.random.choice(length, size=change, replace=False)

    if np.random.rand() < mut:
        new_specie[y, selection] = np.random.randint(0, 256, size=len(selection))
    else:
        new_specie[y, selection] += np.random.randint(-43, 43, size=len(selection))
        new_specie[y, selection] = np.clip(new_specie[y, selection], 0, 255)

    return new_specie
