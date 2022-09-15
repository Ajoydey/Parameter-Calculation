import numpy as np


def corr(img1, img2):
    cor = np.corrcoef(img1.reshape(-1), img2.reshape(-1))[0][1]
    return cor
