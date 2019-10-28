from utils import *
import pandas as p
from sklearn.ensemble import *
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

e = Executor()

abalone = p.read_csv(get_csv_dataset("abalone"))
abalone['Sex'] = abalone['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = abalone.iloc[:, :-1]
y = abalone.iloc[:, -1]


def train_regressors(max_trees_number: int = 50):
    forests = []
    for n_estimators in range(1, max_trees_number + 1):
        print(f"Training a forest with {n_estimators} trees")
        forest = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        forest.fit(X, y)
        forests.append(forest)
    return forests


def cross_validate(regressor: RandomForestRegressor) -> float:
    print(f"Cross-validating the forest with {regressor.n_estimators} trees")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold.split(X, y)
    scores = cross_val_score(regressor, X, y, cv=kfold, scoring="r2")
    return np.average(scores)


def find_min_trees_number(regressors, scores, quality_threshold=0.52) -> int:
    for (forest, score) in zip(regressors, scores):
        forest: RandomForestRegressor
        print(f"The forest with {forest.n_estimators} trees got a score of {score}")
        if score >= quality_threshold:
            return forest.n_estimators
    return -1


print("Training forests...")
regressors = train_regressors()

scores = list(map(cross_validate, regressors))

assert len(scores) == len(regressors), "The number of scores should equal the number of regressors"

e.print_answer("Minimal number of trees", find_min_trees_number(regressors, scores))
