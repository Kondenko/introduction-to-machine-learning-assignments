from collections import defaultdict

import skimage
from matplotlib import pylab
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio
from sklearn.cluster import KMeans
import numpy as np
from numpy import ndarray
from utils import Executor

image = imread("./parrots.jpg")

imgarr: np.ndarray = skimage.img_as_float(image)

X: ndarray = imgarr.reshape((-1, imgarr.shape[2]))

def clusterize(n_clusters=8, show_images=False) -> float:
    clf = KMeans(init="k-means++", random_state=241, n_clusters=n_clusters)
    clf.fit(X)
    # Mean
    X_mean = list(map(lambda p: clf.cluster_centers_[p], clf.labels_))
    mse_mean = mse(np.array(X_mean))
    psnr_mean = peak_signal_noise_ratio(X, np.array(X_mean))
    if show_images:
        image_mean = np.reshape(X_mean, image.shape)
        fig_mean = pylab.figure()
        fig_mean.canvas.set_window_title("Mean")
        pylab.imshow(image_mean)
    # Median
    clusterized_data: dict = defaultdict(list)
    for i in range(0, len(clf.labels_) - 1):
        label = clf.labels_[i]
        clusterized_data[label].append(X[i])
    cluster_medians = {}
    for c in clusterized_data.keys():
        cluster_medians[c] = np.median(clusterized_data[c], 0)
    X_median = list(map(lambda p: cluster_medians[p], clf.labels_))
    mse_median = mse(np.array(X_median))
    psnr_median = peak_signal_noise_ratio(X, np.array(X_median))
    if show_images:
        image_median = np.reshape(X_median, image.shape)
        fig_median = pylab.figure()
        fig_median.canvas.set_window_title("Median")
        pylab.imshow(image_median)
    print(f"{n_clusters} clusters")
    print(f"PSNR for mean color = {psnr_mean}")
    print(f"PSNR for median color = {psnr_median}")
    return max(psnr_mean, psnr_median)


def psnr(mse, max=1):
    return 20 * np.log(max / np.sqrt(mse))


def mse(X_hat, shape: tuple = image.shape):
    return np.sum(sqr_abs(X, X_hat, 0) + sqr_abs(X, X_hat, 1) + sqr_abs(X, X_hat, 2)) / (3 * shape[0] * shape[1])


def sqr_abs(X, X_hat, channel):
    return np.power(np.abs(X[:, channel] - X_hat[:, channel]), 2)


def find_min_num_of_clusters_for_psnr(max_clusters: int = 20, target: int = 20):
    for clusters in range(1, max_clusters):
        psnr = clusterize(clusters)
        if psnr > target:
            print()
            print(f"Reached target PSNR {psnr} with {clusters} clusters")
            return clusters
    print("The algorithm did not converge")
    return None


e = Executor()

min_num_of_clusters = find_min_num_of_clusters_for_psnr()

e.print_answer("Minimal number of clusters", min_num_of_clusters)

pylab.show()
