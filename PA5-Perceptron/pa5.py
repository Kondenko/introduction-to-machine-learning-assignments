import numpy

from utils import *
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

e = Executor()

# Training dataset
dataset_train = pd.read_csv(get_csv_path("perceptron-train"), header=None)
x_train = dataset_train.iloc[:, 1:]
y_train = dataset_train.iloc[:, 0]

# Test dataset
dataset_test = pd.read_csv(get_csv_path("perceptron-test"))
x_test = dataset_test.iloc[:, 1:]
y_test = dataset_test.iloc[:, 0]

# Perceptron

perceptron = Perceptron(random_state=241)


def get_accuracy(x, y):
    y_predicted = perceptron.predict(x)
    return accuracy_score(y, y_predicted)


# Raw data

perceptron.fit(x_train, y_train)

acc = get_accuracy(x_test, y_test)

e.print_answer("Accuracy before scaling", acc, False)

# Scaled data

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

perceptron.fit(x_train_scaled, y_train)

acc_scaled = get_accuracy(x_test_scaled, y_test)

e.print_answer("Accuracy after scaling", acc_scaled, False)

# Finding the accuracy increase

delta = round((acc_scaled - acc), 3)

e.print_answer("Accuracy delta", delta)
