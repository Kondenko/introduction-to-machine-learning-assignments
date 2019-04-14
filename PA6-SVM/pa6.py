from sklearn.svm import SVC
from utils import *
import pandas as pd

e = Executor()

dataset = pd.read_csv(get_csv_dataset("svm-data"), header=None)
x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

clf = SVC(random_state=241, C=100000, kernel='linear')
clf.fit(x, y)

answer = ','.join(str(n + 1) for n in clf.support_)
e.print_answer("Support object indices", answer)
