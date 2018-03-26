import pandas as p
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from utils import *

# Utils

e = Executor()

# Data

classes = ["Class"]
attributes = [
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
]
names = classes + attributes

data_file = get_datasets_folder() + "wine.csv"
dataset = p.read_csv(data_file, header=None, index_col=None, names=names)
target = dataset[classes].values.flatten()  # turn into a 1d array
data = dataset.drop(classes, axis=1)


#   Cross-validation

def optimal_k(x, y):
    opt_k = 0
    max_quality = 0
    generator = KFold(n_splits=5, shuffle=True, random_state=42)  # shuffles the dataset and breaks it into n (5) parts
    classifier = KNeighborsClassifier()
    for __k in range(1, 50):
        classifier.n_neighbors = __k
        qualities = cross_val_score(estimator=classifier, X=x, y=y, cv=generator)
        avg_quality = sum(qualities) / float(len(qualities))
        if avg_quality >= max_quality:
            max_quality = avg_quality
            opt_k = __k
    return [opt_k, max_quality]


e.print_title("Cross-validation classification accuracy (without scaling)")

k_and_quality = optimal_k(data, target)

k = k_and_quality[0]
quality = round2(k_and_quality[1])

e.print_answer("Not scaled - optimal k", k)
e.print_answer("Not scaled - best quality", quality)


e.print_title("Cross-validation classification accuracy (with scaling)")

scaled_data = scale(data)

scaled_k_and_quality = optimal_k(scaled_data, target)

scaled_k = scaled_k_and_quality[0]
scaled_quality = round2(scaled_k_and_quality[1])

e.print_answer("Scaled - optimal k", scaled_k)
e.print_answer("Scaled - best quality", scaled_quality)
