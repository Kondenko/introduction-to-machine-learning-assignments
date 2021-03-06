from math import e

from utils import *
from numpy.core.multiarray import ndarray
from pandas import DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import numpy as np
import pandas as p
import matplotlib.pyplot as plt

plots_dir = os.path.join(os.getcwd(), "plot")
if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)

executor = Executor()

data: DataFrame = p.read_csv(get_csv_dataset("gbm-data"))

array: ndarray = data.values
X: ndarray = array[:, 1:]
y: ndarray = array[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


def train_gb_classifier(learning_rate: float):
    print(f"\n\nTraining a GradientBoostingClassifier with learning rate = {learning_rate}\n")
    n_estimators = 250
    clf = GradientBoostingClassifier(n_estimators=n_estimators, verbose=True, random_state=241, learning_rate=learning_rate)
    clf.fit(X_train, y_train)
    y_pred_train = np.array(list(clf.staged_decision_function(X_train)))
    y_pred_test = np.array(list(clf.staged_decision_function(X_test)))
    y_pred_train_sigmoid = np.array(list(map(sigmoid, y_pred_train)))
    y_pred_test_sigmoid = np.array(list(map(sigmoid, y_pred_test)))
    loss_train: ndarray = np.array(list(map(lambda y_pred: log_loss(y_train, y_pred), y_pred_train_sigmoid)))
    loss_test: ndarray = np.array(list(map(lambda y_pred: log_loss(y_test, y_pred), y_pred_test_sigmoid)))
    plot(learning_rate, loss_test, loss_train)
    iteration = loss_test.argmin() + 1
    loss = loss_test[iteration]
    return loss, iteration


def train_forest_classifier(trees_number: int):
    clf: RandomForestClassifier = RandomForestClassifier(n_estimators=trees_number, random_state=241)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    return log_loss(y_test, y_pred)


def plot(learning_rate, loss_test, loss_train):
    filename = f"{plots_dir}/lr{learning_rate}.png"
    open(filename, "wb+").close()
    fig = plt.figure()
    fig.suptitle(f"Learning rate: {learning_rate}")
    plt.plot(loss_test, 'r', linewidth=2)
    plt.plot(loss_train, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig(filename)


def sigmoid(y_pred):
    return 1. / (1. + np.exp(-y_pred))


rates = [1, 0.5, 0.3, 0.2, 0.1]
results = dict(zip(rates, list(map(train_gb_classifier, rates))))

### 3 ###

executor.print_answer("Overfitting or underfitting", "overfitting")

### 4 ###

loss, iteration = results[0.2]
answer = f"{round2(loss)} {iteration}"
executor.print_answer("Smallest log_loss and it iteration number on the test dataset", answer)

### 5 ###

best_loss, best_iter = min(results.values())
random_forest_loss = train_forest_classifier(trees_number=best_iter)
executor.print_answer("Random forest loss", round2(random_forest_loss))
