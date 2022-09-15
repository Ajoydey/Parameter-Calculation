import numpy as np


def mean(img):
    mean = []
    for i in range(0, 3):
        mean.append(np.mean(img[:, :, i]))
    return np.mean(np.array(mean, dtype=object))


def std(img):
    std = []
    for i in range(0, 3):
        std.append(np.std(img[:, :, i]))
    return np.mean(np.array(std, dtype=object))
