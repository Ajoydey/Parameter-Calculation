from math import log10, sqrt
import numpy as np


def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr, mse
