from math import e

from numpy.core.multiarray import ndarray
from pandas import DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from utils import *
import numpy as np
import pandas as p

executor = Executor()

data: DataFrame = p.read_csv(get_csv_dataset("gbm-data"))

array: ndarray = data.values
X: ndarray = array[:, 1:]
y: ndarray = array[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


def train_classifier(learning_rate: float):
    n_estimators = 250
    clf = GradientBoostingClassifier(n_estimators=n_estimators, verbose=True, random_state=241, learning_rate=learning_rate)
    clf.fit(X_train, y_train)
    y_pred_train = np.array(list(clf.staged_decision_function(X_train)))
    y_pred_test = np.array(list(clf.staged_decision_function(X_test)))
    y_pred_train_sigmoid = np.array(list(map(sigmoid, y_pred_train)))
    y_pred_test_sigmoid = np.array(list(map(sigmoid, y_pred_test)))
    loss_train: ndarray = np.array(list(map(lambda y_pred: log_loss(y_train, y_pred), y_pred_train_sigmoid)))
    loss_test: ndarray = np.array(list(map(lambda y_pred: log_loss(y_test, y_pred), y_pred_test_sigmoid)))
    return loss_train.argmin(), loss_test.argmin()


def sigmoid(y_pred):
    return 1 / (1 + e ** -y_pred)


rates = [1, 0.5, 0.3, 0.2, 0.1]

print(train_classifier(1))
