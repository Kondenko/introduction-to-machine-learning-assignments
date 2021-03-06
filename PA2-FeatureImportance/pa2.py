import pandas as p
from sklearn.tree import DecisionTreeClassifier
from utils import *

e = Executor()

clf = DecisionTreeClassifier(random_state=241)

features = ['Pclass', 'Fare', 'Age', 'Sex']
data = get_titanic_dataset(p)
data = data[p.notna(data['Age'])].replace('male', 0).replace('female', 1)

df_features = data.loc[:, features]
df_target = data['Survived']

clf.fit(df_features, df_target)

title = "Feature importances"


def calculate_importances(classifier):
    importances_dict = dict(zip(df_features.columns, classifier.feature_importances_))
    importances_sorted = sorted(
        importances_dict.iteritems(),
        key=lambda (k, v): (v, k),  # sort by values
        reverse=True
    )
    return "{} {}".format(importances_sorted.pop(0)[0], importances_sorted.pop(0)[0])


e.print_answer(title, calculate_importances(clf))
