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
        return np.array([1 / (1 + np.exp(-(x.T[0] * self.w1 + x.T[1] * self.w2)))]).T

    def gradient_descent(self, x, y, steps=10_000, k=0.1, c=10, regularize=False):
        e = 1e-5
        for _ in range(1, steps):
            delta_w1 = self.w1 + k * 1 / len(y) * np.sum(y[0] * x.T[0].T * (1 - self.predict(x)).T) - (k * c * self.w1 if regularize else 0)
            delta_w2 = self.w2 + k * 1 / len(y) * np.sum(y[0] * x.T[1].T * (1 - self.predict(x)).T) - (k * c * self.w2 if regularize else 0)
            if abs(self.w1 - delta_w1) < e and abs(self.w2 - delta_w2) < e:
                break
            self.w1 += delta_w1
            self.w1 += delta_w2


e.print_title("AUC-ROC w/ and w/o regularization")

lr = LogisticRegression()
lr_reg = LogisticRegression()

lr.gradient_descent(X, Y)
lr_reg.gradient_descent(X, Y, regularize=True)

y_actual = lr.predict(X)
y_actual_reg = lr_reg.predict(X)

score = roc_auc_score(Y, y_actual)
score_reg = roc_auc_score(Y, y_actual_reg)

e.write_to_file("lr_auc_roc", f"{score} {score_reg}")
