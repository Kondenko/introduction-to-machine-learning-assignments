import pandas as p
import numpy as np
from utils import *

ex = Executor("PA1-DataPreparationPandas")

data = p.read_csv(filepath_or_buffer='F:\\Python projects\\projects\\IntroToMachineLearning\\titanic.csv', index_col='PassengerId', engine='python')

header = "Males and females"


def male_and_female_count():
    sex_data = data['Sex'].value_counts()
    answer_tuple = [sex_data['male'], sex_data['female']]
    return "{} {}".format(answer_tuple[0], answer_tuple[1])


ex.execute(header, male_and_female_count)


header = "Survived passengers"


def survived_passengers():
    total = len(data)
    survived = len(data[data['Survived'] == 1])
    return round2(percent(survived, total))


ex.execute(header, survived_passengers)


header = "First class passengers"


def first_class_passengers():
    total = len(data)
    first_class = len(data[data['Pclass'] == 1])
    return round2(percent(first_class, total))


ex.execute(header, first_class_passengers)


header = "Age average and median"


def age_avg_med():
    ages = data['Age']
    answer_tuple = [round2(np.nanmean(ages)), round2(np.nanmedian(ages))]
    return "{} {}".format(answer_tuple[0], answer_tuple[1])


ex.execute(header, age_avg_med)


header = "Pearson correlation"


def pearson_correlation():
    siblings_and_spouses = data['SibSp']
    parents_and_children = data['Parch']
    return round2(siblings_and_spouses.corr(parents_and_children))


ex.execute(header, pearson_correlation)


header = "The most popular feminine name"


def most_popular_feminine_name():
    name = data['Name'][data['Sex'] == 'female'] \
        .str.extract('(\\w+[a|e]\\b)', expand=False) \
        .value_counts() \
        .index[0]
    return name


ex.execute(header, most_popular_feminine_name)
