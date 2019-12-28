from collections import defaultdict

import matplotlib
import skimage
import matplotlib as mpt
from matplotlib import pylab
from skimage.io import imread
from sklearn.cluster import KMeans
import numpy as np
from numpy import ndarray

image = imread("./parrots.jpg")

imgarr: np.ndarray = skimage.img_as_float(image)

X: ndarray = imgarr.reshape((-1, imgarr.shape[2]))  # [:20]  # REMOVE [20]


def clusterize() -> float:
    clf = KMeans(init="k-means++", random_state=241)
    clf.fit(X)
    # Mean
    X_mean = list(map(lambda p: clf.cluster_centers_[p], clf.labels_))
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
    image_median = np.reshape(X_median, image.shape)
    fig_median = pylab.figure()
    fig_median.canvas.set_window_title("Median")
    pylab.imshow(image_median)
    return 0


clusterize()

pylab.show()
