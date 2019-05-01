import pandas as p
from pandas import DataFrame
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from utils import *

col_desc = "FullDescription"
col_salary = "SalaryNormalized"

e = Executor()

tfidf_vectorizer = TfidfVectorizer(min_df=5)

dictVectorizer = DictVectorizer()

ridge = Ridge(alpha=1, random_state=241)

data_train: DataFrame = p.read_csv(get_csv_dataset("salary-train"))
data_test: DataFrame = p.read_csv(get_csv_dataset("salary-test-mini"))


data_train[col_desc] = data_train[col_desc].apply(lambda x: x.lower())
data_train[col_desc] = data_train[col_desc].replace('[^a-zA-Z0-9]', ' ', regex=True)

desc_vectorized_train = tfidf_vectorizer.fit_transform(data_train[col_desc])
desc_vectorized_test = tfidf_vectorizer.transform(data_test[col_desc])


def dict_vectorize(transform_func, data):
    return transform_func(data[['LocationNormalized', 'ContractTime']].to_dict('records'))


def fix_missing_values(column):
    data_train[column].fillna('nan', inplace=True)


fix_missing_values("LocationNormalized")
fix_missing_values("ContractTime")

X_train_categ = dict_vectorize(dictVectorizer.fit_transform, data_train)
X_test_categ = dict_vectorize(dictVectorizer.transform, data_test)

X_train = hstack([desc_vectorized_train, X_train_categ])
y_train = data_train[col_salary]

X_test = hstack([desc_vectorized_test, X_test_categ])


ridge.fit(X_train, y_train)

predictions = ridge.predict(X_test)

answer = ",".join(list(map(lambda n: round2(n), predictions)))

title = "Predicted salary"

e.print_answer(title, answer)