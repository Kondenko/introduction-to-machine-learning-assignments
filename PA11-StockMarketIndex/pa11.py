from pandas import DataFrame

from utils import *
import pandas as p
from sklearn.decomposition import PCA
from numpy import corrcoef, ndarray, iterable

e = Executor()

close_prices: DataFrame = p.read_csv(get_csv_dataset("close_prices"))
names = close_prices.columns.values[1:]
X = close_prices.iloc[:, 1:].T

pca = PCA(n_components=10)
components = pca.fit_transform(X)

def find_min_components_number(dispersion_threshold=0.9):
    sum = 0
    for i in (0, len(pca.explained_variance_ratio_) - 1):
        sum += pca.explained_variance_ratio_[i]
        if sum >= dispersion_threshold:
            return i + 1
    return pca.n_components


e.execute("How many components are needed to explain 90% of dispersion", find_min_components_number)

djia_index: DataFrame = p.read_csv(get_csv_dataset("djia_index"))

corr_x = djia_index.iloc[:, 1:]
corr_y = pca.components_[0].reshape((len(corr_x), 1))
correlation = corrcoef(corr_x.T, corr_y.T)[0][1]

e.print_answer("Pearson correlation between the first component and DJIA", round2(correlation))

index = pca.components_[0].argmax()
company_with_max_weight_name = X[index].argmax()

e.print_answer("Company with max weight", company_with_max_weight_name)
