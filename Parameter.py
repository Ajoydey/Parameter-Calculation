import cv2
import pandas as pd
from ssim import cal_ssim
from psnr import PSNR
from corr import corr
import brisque
import entropy
import mean_std


def referenceParameter(img1, img2):
    ssim_value = cal_ssim(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    psnr_value, mse_value = PSNR(img1, img2)
    corr_value = corr(img1, img2)
    return ssim_value, psnr_value, mse_value, corr_value


def noReferenceParameter(img):
    bris = brisque.BRISQUE()
    bris_score = bris.get_score(img)
    en = entropy.entropyEachChannel(img)
    mean = mean_std.mean(img)
    std = mean_std.std(img)
    return bris_score, en, mean, std


def main():
    hazy = cv2.imread('pics\hazy.jpg')
    dehazed = cv2.imread('pics\dehazed.jpg')
    gt = cv2.imread('pics\gt.jpg')

    brisque_hazy, entropy_hazy, mean_hazy, std_hazy = noReferenceParameter(hazy)

    brisque_dehazed, entropy_dehazed, mean_dehazed, std_dehazed = noReferenceParameter(dehazed)

    brisque_gt, entropy_gt, mean_gt, std_gt = noReferenceParameter(gt)

    df = pd.DataFrame([[brisque_hazy, mean_hazy, std_hazy, entropy_hazy[0], entropy_hazy[1], entropy_hazy[2]], [brisque_dehazed, mean_dehazed, std_dehazed, entropy_dehazed[0], entropy_dehazed[1], entropy_dehazed[2]], [brisque_gt, mean_gt, std_gt, entropy_gt[0], entropy_gt[1], entropy_gt[2]]], index=['Hazy pic', 'Dehazed pic', 'Ground truth'], columns=['BRISQUE_VALUE', 'MEAN', 'STANDARD DEVIATION', 'ENTROPY_B', 'ENTROPY_G', 'ENTROPY_R'])

    print(df)

    ssim, psnr, mse, corr = referenceParameter(hazy, gt)
    ssim1, psnr1, mse1, corr1 = referenceParameter(dehazed, gt)

    df2 = pd.DataFrame([[ssim, psnr, mse, corr], [ssim1, psnr1, mse1, corr1]], index=['Hazy~Ground truth', 'Dehazed~Ground truth'], columns=['SSIM', 'PSNR', 'MSE', 'CORRELATION'])

    print(df2)


if __name__ == "__main__":
    main()
