from collections import defaultdict

import matplotlib
import skimage
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
    pylab.imshow(image_mean)
    pylab.show()
    return 0

clusterize()
