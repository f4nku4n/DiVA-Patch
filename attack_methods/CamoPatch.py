import cv2
import math
import numpy as np
from tqdm import tqdm
from attack_methods.base import Attacker
from attack_methods.utils import l2_compute, sh_selection


class CamoPatch(Attacker):
    def __init__(self, params, loss_function, max_query=10000, setting='realistic'):
        super().__init__(max_query)
        self.loss_function = loss_function
        self.params = params
        self.loss_function.bind_base_image(params["x"])
        self.setting = setting

    def completion_procedure(self, is_success, x_adv, queries, loc, patch):
        data = {
            "adversarial": is_success,
            "queries": queries,
            "first_success_query": self.first_success_query,
            "loc": loc,
            "patch": patch,
            "final_prediction": self.loss_function.get_label(x_adv),
            "process": self.process
        }

        np.save(self.params["save_directory"], data, allow_pickle=True)

    def run(self):
        if self.setting == 'realistic':
            min_v, max_v = 0., 255.
        else:
            min_v, max_v = 0., 1.
        x = self.params["x"]

        c, h, w = self.params["c"], self.params["h"], self.params["w"]
        eps = self.params["eps"]
        s = int(math.ceil(eps ** .5))

        if self.setting == 'realistic':
            patch_geno = np.random.randint(0, 256, size=(self.params["N"], 7))
            loc = [np.random.randint(h - s), np.random.randint(w - s)]
        else:
            patch_geno = np.random.rand(self.params["N"], 7)
            loc = np.random.randint(h - s, size=2)

        patch = render(patch_geno, s, s, setting=self.setting)

        update_loc_period = self.params["update_loc_period"]

        x_adv = compose_image(x, patch, loc, (min_v, max_v))
        is_success, loss = self.loss_function.evaluate_patch(patch, loc, clip_bounds=(min_v, max_v))
        self.count_query(is_success)

        l2_curr = l2_compute(
            adv_patch=patch,
            orig_patch=x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy(),
            integer=(self.setting == 'realistic')
        )
        self.process.append([is_success, loc, patch_geno, l2_curr, loss])

        patch_counter = 0

        for it in tqdm(range(1, self.max_query)):
            patch_counter += 1
            if patch_counter < update_loc_period:
                patch_new_geno = mutate(patch_geno, self.params["mut"], setting=self.setting)
                patch_new = render(patch_new_geno, s, s, setting=self.setting)
                # evaluate new solutions
                is_success_new, loss_new = self.loss_function.evaluate_patch(
                    patch_new, loc,
                    clip_bounds=(min_v, max_v)
                )
                self.count_query(is_success_new)

                orig_patch = x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy()

                l2_new = l2_compute(
                    adv_patch=patch_new,
                    orig_patch=orig_patch,
                    integer=(self.setting == 'realistic')
                )

                if is_success is True and is_success_new is True:
                    if l2_new < l2_curr:
                        loss = loss_new
                        is_success = is_success_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = compose_image(x, patch_new, loc, (min_v, max_v))
                        l2_curr = l2_new
                else:
                    if loss_new < loss:  # minimization
                        loss = loss_new
                        is_success = is_success_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = compose_image(x, patch_new, loc, (min_v, max_v))
                        l2_curr = l2_new

            else:
                patch_counter = 0

                # location update
                sh_i = int(max(sh_selection(self.max_query, it) * h, 0))
                sw_i = int(max(sh_selection(self.max_query, it) * w, 0))
                loc_new = loc.copy()
                loc_new = update_location(loc_new, sh_i, sw_i, h, w, s, setting=self.setting)

                # evaluate new solution
                is_success_new, loss_new = self.loss_function.evaluate_patch(
                    patch, loc_new, clip_bounds=(min_v, max_v)
                )
                self.count_query(is_success_new)

                orig_patch_new = x[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :].copy()
                l2_new = l2_compute(
                    adv_patch=patch,
                    orig_patch=orig_patch_new,
                    integer=(self.setting == 'realistic')
                )

                if is_success is True and is_success_new is True:
                    if l2_new < l2_curr:
                        loss = loss_new
                        is_success = is_success_new
                        loc = loc_new

                        x_adv = compose_image(x, patch, loc_new, (min_v, max_v))
                        l2_curr = l2_new
                else:
                    diff = loss_new - loss
                    curr_temp = self.params["temp"] / (it + 1)
                    metropolis = math.exp(-diff / curr_temp)

                    if loss_new < loss or np.random.rand() < metropolis:  # minimization # first check
                        loss = loss_new
                        is_success = is_success_new
                        loc = loc_new
                        x_adv = compose_image(x, patch, loc_new, (min_v, max_v))
                        l2_curr = l2_new
            self.process.append([is_success, loc, patch_geno, l2_curr, loss])

        self.completion_procedure(is_success, x_adv, self.n_query, loc, patch)
        return

###################################################### Utilities #######################################################
def compose_image(image, patch, location, clip_bounds):
    image_adv = image.copy()
    x, y = int(location[0]), int(location[1])
    patch_h, patch_w = patch.shape[:2]
    image_adv[x:x + patch_h, y:y + patch_w, :] = patch
    return np.clip(image_adv, clip_bounds[0], clip_bounds[1])


def update_location(loc_new, h_i, w_i, h, w, s, setting='realistic'):
    if setting == 'realistic':
        loc_new[0] += np.random.randint(low=-h_i, high=h_i + 1)
        loc_new[0] = np.clip(loc_new[0], 0, h - s)
        loc_new[1] += np.random.randint(low=-w_i, high=w_i + 1)
        loc_new[1] = np.clip(loc_new[1], 0, w - s)
    else:
        loc_new += np.random.randint(low=-h_i, high=h_i + 1, size=(2,))
        loc_new = np.clip(loc_new, 0, h - s)
    return loc_new


def render(x, h, w, setting='realistic'):
    phenotype = np.ones((h, w, 3)) # load a white patch w*w
    if setting == 'realistic':
        phenotype = np.ones((h, w, 3), dtype=np.uint8) * 255  # load a white patch w * w
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        if setting == 'realistic':
            cv2.circle(
                overlay,
                center=(int(row[1] * h / 255), int(row[0] * w / 255)),
                radius=int(row[2] / 255 * 2 * radius_avg),
                color=(int(row[3]), int(row[4]), int(row[5])),
                thickness=-1,
            )
            alpha = row[6] / 255.
        else:
            cv2.circle(
                overlay,
                center=(int(row[1] * h), int(row[0] * w)),
                radius=int(row[2] * radius_avg),
                color=(int(row[3] * 255), int(row[4] * 255), int(row[5] * 255)),
                thickness=-1,
            )
            alpha = row[6]
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)
    if setting != 'realistic':
        phenotype = phenotype / 255.
    return phenotype

def mutate(soln, mut=0.3, setting='realistic', rng=None):
    rng = np.random if rng is None else rng
    new_specie = soln.copy()

    # Randomization for Evolution
    genes = soln.shape[0]
    length = soln.shape[1]
    y = rng.randint(0, genes)
    change = rng.randint(0, length + 1)

    selection = rng.choice(length, size=change, replace=False)

    if rng.rand() < mut:
        if setting == 'realistic':
            new_specie[y, selection] = rng.randint(0, 256, size=len(selection))
        else:
            new_specie[y, selection] = rng.rand(len(selection))
    else:
        if setting == 'realistic':
            new_specie[y, selection] += rng.randint(-43, 43, size=len(selection))
            new_specie[y, selection] = np.clip(new_specie[y, selection], 0, 255)
        else:
            new_specie[y, selection] += (rng.rand(len(selection)) - 0.5) / 3
            new_specie[y, selection] = np.clip(new_specie[y, selection], 0, 1)
    return new_specie
