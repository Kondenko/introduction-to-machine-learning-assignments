import math

import pandas as p
import numpy as np
import utils as ut
from sklearn.metrics import roc_auc_score

e = ut.Executor()

dataset = p.read_csv(ut.get_csv_path("data-logistic"))
X = dataset.values[:, 1:]
Y = dataset.values[:, :1]


class LogisticRegression:

    def __init__(self):
        self.w1 = 0
        self.w2 = 0

    def predict(self, x):
        return np.array([1 / (1 + np.exp(-(x[:, 0] * self.w1 + x[:, 1] * self.w2)))]).T

    def gradient_descent(self, x, y, steps=10_000, k=0.1, c=10, regularize=False):
        e = 1e-5

        for i in range(1, steps):
            x1 = x[:, 0]
            x2 = x[:, 1]

            def sigmoid_exp():
                return 1 - 1 / (1 + np.exp(-y * np.array([(self.w1 * x1 + self.w2 * x2)]).T))

            new_w1 = self.w1 + k * np.mean((y.T * x1).T * sigmoid_exp()) - (k * c * self.w1 if regularize else 0)
            new_w2 = self.w2 + k * np.mean((y.T * x2).T * sigmoid_exp()) - (k * c * self.w2 if regularize else 0)

            if self.__distance__([self.w1, self.w2], [new_w1, new_w2]) < e:
                break

            self.w1 = new_w1
            self.w2 = new_w2

    def __distance__(self, w, w_new):
        return np.linalg.norm(np.array([w]) - np.array([w_new]))


e.print_title("AUC-ROC w/ and w/o regularization")

lr = LogisticRegression()
lr_reg = LogisticRegression()

lr.gradient_descent(X, Y)
lr_reg.gradient_descent(X, Y, regularize=True)

y_actual = lr.predict(X)
y_actual_reg = lr_reg.predict(X)

score = round(roc_auc_score(Y, y_actual), 3)
score_reg = round(roc_auc_score(Y, y_actual_reg), 3)

assert score == 0.927 and score_reg == 0.936

e.write_to_file("lr_auc_roc", f"{score} {score_reg}")
