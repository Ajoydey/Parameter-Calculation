import cv2
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
    hazy = cv2.imread('pics/3I.jpg')
    dehazed = cv2.imread('pics/3J.jpg')

    brisque, entropy, mean, std = noReferenceParameter(hazy)
    print("No reference parameter values for hazy image:\n")
    print(f"BRISQUE value is {brisque}")
    print(f"Entropy value channelwise {entropy}")
    print(f"Mean values are {mean}")
    print(f"Standard deviation values are {std}")

    brisque, entropy, mean, std = noReferenceParameter(dehazed)
    print("\nNo reference parameter values for dehazed image:\n")
    print(f"BRISQUE value is {brisque}")
    print(f"Entropy value channelwise {entropy}")
    print(f"Mean values are {mean}")
    print(f"Standard deviation values are {std}")

    ssim, psnr, mse, corr = referenceParameter(hazy, dehazed)
    print("\nReference parameter values:\n")
    print(f"SSIM value is {ssim}")
    print(f"PSNR value is {psnr} dB")
    print(f"MSE value is {mse} dB")
    print(f"Correlation value is {corr} dB")


if __name__ == "__main__":
    main()
