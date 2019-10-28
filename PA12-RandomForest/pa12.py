from utils import *
import pandas as p

e = Executor()

abalone = p.read_csv(get_csv_dataset("abalone"))
abalone['Sex'] = abalone['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = abalone.iloc[:, :-1]
y = abalone.iloc[:, -1]

print(X.head())
print(y.head())