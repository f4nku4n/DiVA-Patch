import numpy as np
import cv2

def render(x, w):
    phenotype = np.ones((w, w, 3))
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        cv2.circle(
            overlay,
            center=(int(row[1] * w), int(row[0] * w)),
            radius=int(row[2] * radius_avg),
            color=(int(row[3] * 255), int(row[4] * 255), int(row[5] * 255)),
            thickness=-1,
        )
        alpha = row[6]
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)

    return phenotype/255.


class Individual:
    def __init__(self, N=100, patch_size=[40, 40]):
        self.location = None
        self.patch_geno = None
        self.patch = None
        self.N = N
        self.s = patch_size
        self.default_patch = patch_size
        self.success_attack = None
        self.improved_l2 = False
        self.l2 = None
        self.ssim_patch = None
        self.loss = None
        self.loss_init = None
        self.aging = 0
        self.best_config = {}

    def update(self):
        if self.best_config['success_attack'] is True and self.success_attack is True:
            if self.l2 < self.best_config['l2']:
                self.best_config['location'] = self.location
                self.best_config['l2'], self.best_config['loss'] = self.l2, self.loss
                self.best_config['patch_geno'], self.best_config['patch'] = self.patch_geno, self.patch
                self.best_config['patch_size'] = self.s
        elif self.best_config['success_attack'] is False and self.success_attack is True:
            self.best_config['location'] = self.location
            self.best_config['success_attack'] = True
            self.best_config['l2'], self.best_config['loss'] = self.l2, self.loss
            self.best_config['patch_geno'], self.best_config['patch'] = self.patch_geno, self.patch
            self.best_config['patch_size'] = self.s
        elif self.best_config['success_attack'] is False and self.success_attack is False:
            if self.loss < self.best_config['loss']:
                self.best_config['location'] = self.location
                self.best_config['l2'], self.best_config['loss'] = self.l2, self.loss
                self.best_config['patch_geno'], self.best_config['patch'] = self.patch_geno, self.patch
                self.best_config['patch_size'] = self.s

    def save(self):
        self.best_config['location'] = self.location.copy()
        self.best_config['patch_geno'] = self.patch_geno.copy()
        self.best_config['patch'] = self.patch.copy()
        self.best_config['patch_size'] = self.s.copy()
        self.best_config['success_attack'] = self.success_attack
        self.best_config['l2'] = self.l2
        self.best_config['loss'] = self.loss

    def update_patch_size(self, img_h, img_w):
        self.s = [min(self.default_patch[0], img_h - self.location[0]), min(self.default_patch[1], img_w - self.location[1])]

    def rand(self, img_h, img_w, lx=None, ux=None, ly=None, uy=None):
        if lx is None and ux is None and ly is None and uy is None:
            lx, ux = 0, img_h
            ly, uy = 0, img_w
        else:
            ux = min(ux, img_h)
            uy = min(uy, img_w)
        self.rand_loc(lx=lx, ux=ux, ly=ly, uy=uy)
        self.rand_patch()

    def rand_loc(self, lx, ux, ly, uy):
        self.location = [np.random.randint(lx, ux), np.random.randint(ly, uy)]

    def rand_patch(self):
        self.patch_geno = np.random.rand(self.N, 7)
        self.patch = render(self.patch_geno, self.s[0])


