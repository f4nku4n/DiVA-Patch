import cv2
import numpy as np
from tqdm import tqdm
from attack_methods.base import Attacker
from attack_methods.utils import l2_compute
from attack_methods.CamoPatch import render, mutate

class PatchRS(Attacker):
    def __init__(self, img_cls, loss_function, p_init=0.4, patch_size=(40, 40),
                 update_loc_period=4, max_query=10000, setting='realistic'):
        super().__init__(max_query=max_query)
        self.img_cls = img_cls
        self.loss_function = loss_function
        self.loss_function.bind_base_image(img_cls)

        self.p_init = p_init

        self.patch_size = patch_size
        self.N = 100
        self.update_loc_period = update_loc_period
        self.rescale_schedule = True

        self.init_patches = 'random_squares'
        self.setting = setting

    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.max_query * 10000)

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init

        return p

    def sh_selection(self, it):
        """ schedule to decrease the parameter p """

        t = max((float(self.max_query - it) / self.max_query - .0) ** 1., 0) * .75

        return t

    def get_init_patch(self, s):
        if self.setting == 'realistic':
            patch_geno = np.random.randint(0, 256, size=(self.N, 7))
        else:
            patch_geno = np.random.rand(self.N, 7)  # Difference
        patch = render(patch_geno, s[0], s[1])
        return patch_geno, patch

    def run(self):
        H, W, C = self.img_cls.shape
        s = self.patch_size

        best_loc = [np.random.randint(0, H - s[0]), np.random.randint(0, W - s[1])]

        best_patch_geno, best_patch = self.get_init_patch(s)

        loc_t = abs(self.update_loc_period)
        assert loc_t > 1

        best_adversarial, best_loss = self.loss_function.evaluate_patch(
            best_patch, best_loc
        )
        best_l2 = l2_compute(
            adv_patch=best_patch,
            orig_patch=self.img_cls[best_loc[0]: best_loc[0] + s[0], best_loc[1]: best_loc[1] + s[1], :].copy(),
            integer=(self.setting == 'realistic')
        )
        self.count_query(best_adversarial)
        self.process.append([self.n_query, best_adversarial, best_loc, best_patch, best_l2, best_loss])

        for it in tqdm(range(1, self.max_query)):
            s_it = int(max(self.p_selection(it) ** .5 * s[0], 1))

            ## Sample update
            new_patch_geno = best_patch_geno.copy()
            new_patch = best_patch.copy()
            new_loc = best_loc.copy()

            sh_it = int(max(self.sh_selection(it) * H, 0))
            sw_it = int(max(self.sh_selection(it) * W, 0))
            update_loc = (it % loc_t == 0) and (sh_it > 0) and (sw_it > 0)
            if self.update_loc_period < 0 < sh_it and sw_it > 0:
                update_loc = not update_loc
            update_patch = not update_loc

            if update_patch:
                # update patch
                if s_it > 1:
                    new_patch_geno = mutate(best_patch_geno, mut=1.1, setting=self.setting)
                else:
                    new_patch_geno = mutate(best_patch_geno, mut=-0.1, setting=self.setting)
                new_patch = render(new_patch_geno, s[0], s[1], setting=self.setting)

            if update_loc:
                new_loc[0] = np.clip(new_loc[0] + np.random.randint(-sh_it, sh_it + 1), 0, H - s[0])
                new_loc[1] = np.clip(new_loc[1] + np.random.randint(-sw_it, sw_it + 1), 0, W - s[1])

            new_adversarial, new_loss = self.loss_function.evaluate_patch(
                new_patch, new_loc
            )
            new_l2 = l2_compute(
                new_patch,
                self.img_cls[new_loc[0]: new_loc[0] + s[0], new_loc[1]: new_loc[1] + s[1], :].copy(),
                integer=(self.setting == 'realistic')
            )
            self.count_query(new_adversarial)

            if is_better(new_adversarial, best_adversarial, new_loss, best_loss, new_l2, best_l2):
                best_adversarial = new_adversarial
                best_loss = new_loss
                best_patch_geno = new_patch_geno
                best_patch = new_patch
                best_loc = new_loc
                best_l2 = new_l2

            self.process.append([self.n_query, best_adversarial, best_loc, best_patch, best_l2, best_loss])


def is_better(new_adversarial, adversarial, new_loss, loss, new_l2, l2):
    if not adversarial and new_adversarial:
        return True
    if new_adversarial and adversarial:
        return new_l2 < l2
    if not new_adversarial and not adversarial:
        return new_loss < loss
    return False
