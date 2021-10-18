import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.feature import plot_matches

FIGSIZE = (15, 10)
COLUMNS = 3
ROWS = 2


def plot_collage(imgs, columns=COLUMNS, rows=ROWS, figsize=FIGSIZE, title=None):
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
        plt.axis('off')

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i - 1], interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_keypoints(img, keypoints):
    plt.figure(figsize=FIGSIZE)
    plt.imshow(img)
    plt.axis('off')
    plt.scatter(keypoints[:, 1], keypoints[:, 0], facecolors='none', edgecolors='r')
    plt.show()


def plot_inliers(src, dest, src_keypoints, dest_keypoints, matches):
    plt.figure(figsize=FIGSIZE)
    ax = plt.axes()
    ax.axis("off")
    ax.set_title(f"Inlier correspondences: {len(matches)} points matched")
    plot_matches(ax, src, dest, src_keypoints, dest_keypoints,
                 matches)
    plt.show()


def plot_warps(corners, output_shape=None, min_coords=None, max_coords=None, img=None):
    np.random.seed(0)
    if output_shape is None:
        plt.figure(figsize=FIGSIZE)
    else:
        plt.figure(figsize=(15, 5))
    ax = plt.axes()

    for coords in corners:
        ax.add_patch(Polygon(coords, closed=True, fill=False, color=np.random.rand(3)))

    if max_coords is not None:
        plt.xlim(min_coords[0], max_coords[0])
        plt.ylim(max_coords[1], min_coords[1])

    if output_shape is not None:
        plt.xlim(0, output_shape[1])
        plt.ylim(output_shape[0], 0)

    if img is not None:
        plt.imshow(img)

    plt.title('Border visualization')
    plt.show()


def plot_gauss(img, merged_img):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title('Input image')
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('Merged image')
    plt.axis('off')
    plt.imshow(merged_img)
    plt.show()


def plot_result(result):
    plt.figure(figsize=FIGSIZE)
    plt.imshow(result)
    plt.axis('off')
    plt.show()
