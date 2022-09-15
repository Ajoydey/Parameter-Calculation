import numpy as np
import cv2


def calcEntropy(img):
    entropy = []

    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(np.array(entropy, dtype=object))
    return sum_en[0]


def entropyEachChannel(img):
    en = []
    for i in range(0, 3):
        en.append(calcEntropy(img[:, :, i]))
    return en
