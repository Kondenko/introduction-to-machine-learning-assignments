from utils import *
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

# Utils

e = Executor()

# Dataset

dataset = load_boston()
data = dataset.data
target = dataset.target

data = scale(data)


def optimal_p(x, y):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsRegressor
    from numpy import linspace
    opt_p = 0
    max_quality = 0
    generator = KFold(n_splits=5, shuffle=True, random_state=42)
    regressor = KNeighborsRegressor(5, weights="distance")
    linspace = linspace(start=1, stop=10, num=200)
    for __p in linspace:
        regressor.p = __p
        qualities = cross_val_score(estimator=regressor, X=x, y=y, cv=generator, scoring="neg_mean_squared_error")
        avg_quality = sum(qualities) / float(len(qualities))
        if avg_quality >= max_quality or max_quality == 0:
            max_quality = avg_quality
            opt_p = __p
    return opt_p


e.print_answer("Optimal P", optimal_p(data, target))
