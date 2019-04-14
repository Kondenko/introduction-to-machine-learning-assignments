from sklearn.metrics import *

import pandas as p
import utils as u

e = u.Executor()

clf = p.read_csv(u.get_csv_dataset("classification"))
true = clf.T.values[0]
pred = clf.T.values[1]

scores = p.read_csv(u.get_csv_dataset("scores"))

###

title = "TP_FP_FN_TN"


def calculate_errors() -> list:
    """
    TP, FP, FN and TN for a table of expected and actual classification results
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for true, pred in clf.values:
        if true == 1:
            if pred == 1:
                tp += 1
            else:
                fp += 1
        else:
            if pred == 0:
                tn += 1
            else:
                fn += 1
    return [tp, fp, fn, tn]


errors = calculate_errors()
answer = u.join(errors)

e.print_answer(title, answer)

###

title = "Primary metrics"

accuracy = accuracy_score(true, pred)
precision = precision_score(true, pred)
recall = recall_score(true, pred)
f1 = f1_score(true, pred)

answer = u.join([accuracy, precision, recall, f1], mapper=lambda s: str(u.round2(s)))

e.print_answer(title, answer)
