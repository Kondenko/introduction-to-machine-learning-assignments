import pandas as p
from pandas import DataFrame
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import *

col_desc = "FullDescription"

e = Executor()

data_train: DataFrame = p.read_csv(get_csv_dataset("salary-train"))
data_test: DataFrame = p.read_csv(get_csv_dataset("salary-test-mini"))

tfidfVectorizer = TfidfVectorizer(min_df=5)

dictVectorizer = DictVectorizer()

data_train[col_desc] = data_train[col_desc].apply(lambda x: x.lower())
data_train[col_desc] = data_train[col_desc].replace('[^a-zA-Z0-9]', ' ', regex=True)

desc_vectorized = tfidfVectorizer.fit_transform(data_train[col_desc])


def dict_vectorize(transform_func, data):
    return transform_func(data[['LocationNormalized', 'ContractTime']].to_dict('records'))


def fix_missing_values(column):
    data_train[column].fillna('nan', inplace=True)


X_train_categ = dict_vectorize(dictVectorizer.fit_transform, data_train)
X_test_categ = dict_vectorize(dictVectorizer.transform, data_test)

fix_missing_values("LocationNormalized")
fix_missing_values("ContractTime")

x = hstack([desc_vectorized, X_train_categ])

print(data_train.sample(5))
