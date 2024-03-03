import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

from numpy import random


def visualize_pics(ims, n=480, shuffle=False, source='wiki_crop_all/'):
    if shuffle:
        random.shuffle(ims)

    n = n // 3 * 3

    plt.figure(figsize=(10, n // 3 * 2))

    for i, img in enumerate(ims[:n]):
        plt.subplot(n // 3, 3, i + 1)

        plt.imshow(mpimg.imread(source + img))

    plt.show()


def visualize_pics_boxes(df, n=480):
    n = n // 3 * 3

    plt.figure(figsize=(10, n // 3 * 2))

    for i, row in df.iloc[:n].iterrows():
        img, box = row
        plt.subplot(n // 3, 3, i + 1)

        plt.imshow(mpimg.imread('wiki_crop_all/' + img), cmap='gray')
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))

    plt.show()
