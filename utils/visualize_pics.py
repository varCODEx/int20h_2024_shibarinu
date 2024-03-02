import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def visualize_pics(ims, n=480):
    n = n // 3 * 3

    plt.figure(figsize=(10, n // 3 * 2))

    for i, img in enumerate(ims[:n]):
        plt.subplot(n // 3, 3, i + 1)

        plt.imshow(mpimg.imread('wiki_crop_all/' + img))

    plt.show()
