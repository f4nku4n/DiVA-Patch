import numpy as np

def l2_compute(adv_patch, orig_patch, integer=False):
    assert adv_patch.shape == orig_patch.shape
    if integer:
        return np.sum((adv_patch / 255. - orig_patch / 255.) ** 2) ** 0.5
    else:
        return np.sum((adv_patch - orig_patch) ** 2) ** 0.5

def sh_selection(n_queries, it):
    """ schedule to decrease the parameter p """
    t = max((float(n_queries - it) / n_queries - .0) ** 1., 0) * .75
    return t
