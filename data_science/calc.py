def report_to_confusion_matrix(data, penalty):
    tp = data['positive']['support'] * data['positive']['recall']
    fn = data['positive']['support'] - tp
    tn = data['negative']['support'] * data['negative']['recall']
    fp = data['negative']['support'] - tn

    total = tp + fn + fp + tn
    support = data['positive']['support'] + data['negative']['support']

    if abs(total - support) > 1:
        print("########## double check ########")

    gain = (tp + tn) * penalty[0]
    loss = (fn + fp) * penalty[1]

    matrix = {
        'TP': '{0:.3g}'.format(tp),
        'FN': '{0:.3g}'.format(fn),
        'FP': '{0:.3g}'.format(fp),
        'TN': '{0:.3g}'.format(tn),
        'Penalty': [gain, loss, gain - loss]
    }

    return matrix


def report_from_confusion_matrix(confusion_matrix, penalty):
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    gain = (TP + TN) * penalty[0]
    loss = (FN + FP) * penalty[1]

    score = {
        'precision/PPV': '{0:.3g}'.format(precision),
        'recall/sensitivity/TPR': '{0:.3g}'.format(recall),
        'f1_score': '{0:.3g}'.format(f1_score),
        'accuracy': '{0:.3g}'.format(accuracy),
        'Penalty': [gain, loss, gain - loss]
    }

    return score


"""
                    precision   recall  f1-score    support
            0       0.86      0.91      0.88        66
            1       0.82      0.72      0.77        37

    accuracy                           0.84       103
   macro avg       0.84      0.81      0.83       103
weighted avg       0.84      0.84      0.84       103
"""

data = {'positive': {}, 'negative': {}}
data['positive']['precision'], data['positive']['recall'], data['positive']['f1-score'], data['positive']['support'] = 0.86, 0.91, 0.88, 66
data['negative']['precision'], data['negative']['recall'], data['negative']['f1-score'], data['negative']['support'] = 0.82, 0.72, 0.77, 37
penalty = [1, 1]

# report to confusion matrix
matrix = report_to_confusion_matrix(data, penalty)

print("\n\n------------------ confusion matrix ----------------")
print(matrix)

# confusion matrix to report
confusion_matrix = [[60, 6],
                    [9, 28]]

penalty = [1, 1]

report = report_from_confusion_matrix(confusion_matrix, penalty)
print("\n\n------------------ classification report ----------------")
print(report)
