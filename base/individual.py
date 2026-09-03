import cv2
import numpy as np

class Individual:
    def __init__(self, N=100, patch_size=(40, 40), setting='realistic'):
        self.location = None
        self.patch_geno = None
        self.patch = None
        self.N = N
        self.s = patch_size
        self.success_attack = None
        self.l2 = None
        self.loss = None
        self.setting = setting

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
        if self.setting == 'realistic':
            self.patch_geno = np.random.randint(0, 256, size=(self.N, 7))
        else:
            self.patch_geno = np.random.rand(self.N, 7)
        self.patch = render(self.patch_geno, self.s[0], self.s[1], self.setting)

def render(x, h, w, setting='realistic'):
    phenotype = np.ones((w, w, 3))
    if setting == 'realistic':
        phenotype = np.ones((h, w, 3), dtype=np.uint8) * 255
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        if setting == 'realistic':
            cv2.circle(
                overlay,
                center=(int(row[1] * h/255), int(row[0] * w/255)),
                radius=int(row[2]/255 * 2 * radius_avg),
                color=(int(row[3]), int(row[4]), int(row[5])),
                thickness=-1,
            )
            alpha = row[6] / 255.
        else:
            cv2.circle(
                overlay,
                center=(int(row[1] * w), int(row[0] * w)),
                radius=int(row[2] * radius_avg),
                color=(int(row[3] * 255), int(row[4] * 255), int(row[5] * 255)),
                thickness=-1,
            )
            alpha = row[6]
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)
    if setting != 'realistic':
        phenotype = phenotype / 255.
    return phenotype
