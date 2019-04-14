from sklearn.metrics import *

import pandas as p
import numpy as np
import utils as u

e = u.Executor()

clf = p.read_csv(u.get_csv_dataset("classification"))
clf_true = clf.T.values[0]
clf_pred = clf.T.values[1]

scores = p.read_csv(u.get_csv_dataset("scores"))
score_true = scores.values[:, 0]
preds = scores.values.T[1:, :]

classifier_names = scores.columns.values.tolist()[1:]

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

accuracy = accuracy_score(clf_true, clf_pred)
precision = precision_score(clf_true, clf_pred)
recall = recall_score(clf_true, clf_pred)
f1 = f1_score(clf_true, clf_pred)

answer = u.join([accuracy, precision, recall, f1], mapper=lambda s: str(u.round2(s)))

e.print_answer(title, answer)

### Best AUC-ROC

title = "The best classificator"

scores['score_logreg'] = scores['score_logreg'].apply(lambda i: i >= 0.5)
scores['score_svm'] = scores['score_svm'].apply(lambda i: i >= 0)
scores['score_knn'] = scores['score_knn'].apply(lambda i: bool(round(i)))
scores['score_tree'] = scores['score_tree'].apply(lambda i: i >= 0.5)

results: tuple = tuple(zip(classifier_names, list(map(lambda pred: roc_auc_score(score_true, pred), preds))))
best_classificator: tuple = max(results)

e.print_answer(title, best_classificator[0])

## Max precision with the given recall

title = "Max precision with the given recall"

recall_threshold = 0.7

recall_curve_values = list(map(lambda pred: precision_recall_curve(score_true, pred), preds))

def get_accuracy(pres_rec_thres: tuple):
    df: p.DataFrame = p.DataFrame(pres_rec_thres, ["precision", "recall", "threshold"]).T
    filtered: p.DataFrame = df.loc[df["recall"] >= recall_threshold]
    return filtered.loc[filtered["precision"].idxmax].precision

accuracies = [get_accuracy(c) for (c) in recall_curve_values]

clf_accuracies = dict(zip(classifier_names, accuracies))

answer = max(clf_accuracies, key=lambda c_a: c_a[1])

e.print_answer(title, answer)
