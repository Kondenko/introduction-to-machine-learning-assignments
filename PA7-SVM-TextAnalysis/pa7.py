import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC

from utils import *

e = Executor()
model_file_name = "svm"
random_state = 241

dataset = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = dataset.data
Y = dataset.target


def find_best_c(x, y):
    param = 'C'
    c_set = {param: np.power(float(10), np.arange(-5, 6))}
    generator = KFold(n_splits=5, shuffle=True, random_state=random_state)
    classifier = SVC(kernel='linear', random_state=random_state, verbose=True)
    param_chooser = GridSearchCV(classifier, c_set, scoring='accuracy', cv=generator)
    param_chooser.fit(x, y)
    return param_chooser.best_params_[param]


def get_top_10_words(feature_names, coefs):
    coefs_abs = list()
    for i in range(coefs.nnz):
        coefs_abs.append((coefs.indices[i], abs(coefs.data[i])))
    coefs_abs.sort(key=lambda (k, v): (v, k), reverse=True)
    return ','.join(sorted(map(lambda item: str(feature_names[item[0]]), coefs_abs[:10])))


vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

c = find_best_c(X_tfidf, Y)

svm = SVC(C=c, kernel='linear', random_state=random_state, verbose=True)
svm.fit(X_tfidf, Y)

e.print_answer("Top 10 words", get_top_10_words(vectorizer.get_feature_names(), svm.coef_), svm)