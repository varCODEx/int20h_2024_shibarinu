import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from numpy import random


def visualize_pics(ims, n=480, shuffle=False):
    if shuffle:
        random.shuffle(ims)

    n = n // 3 * 3

    plt.figure(figsize=(10, n // 3 * 2))

    for i, img in enumerate(ims[:n]):
        plt.subplot(n // 3, 3, i + 1)

        plt.imshow(mpimg.imread('wiki_crop_all/' + img))

    plt.show()
