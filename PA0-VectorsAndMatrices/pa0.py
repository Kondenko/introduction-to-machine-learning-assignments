import numpy as np
from utils import *

matrix_x = np.random.normal(1, 10, (1000, 50))

print matrix_x

avg = np.mean(matrix_x, axis=0)
std_deviation = np.std(matrix_x, axis=0)

matrix_norm = (matrix_x - avg) / std_deviation

print_title("Norm matrix")

print matrix_norm

print_title("Matrix with lines more than 10")

matrix_z = np.array([[4, 5, 0],
                     [1, 9, 3],
                     [5, 1, 1],
                     [3, 3, 3],
                     [9, 9, 9],
                     [4, 7, 1]])

line_sums = np.sum(matrix_z, axis=1)

print   np.flatnonzero(line_sums > 10)

print_title("Eye-matrix")

matrix_eye_a = np.eye(3)
matrix_eye_b = np.eye(3)
matrix_eye_sum = np.vstack((matrix_eye_a, matrix_eye_b))

print matrix_eye_a
print " + "
print matrix_eye_b
print " = "
print matrix_eye_sum
