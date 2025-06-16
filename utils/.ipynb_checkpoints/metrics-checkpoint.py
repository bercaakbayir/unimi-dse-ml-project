import numpy as np
import pandas as pd
def true_positive(y_true, y_pred, positive_label='good'):
    return np.sum((y_pred == positive_label) & (y_true == positive_label))

def false_positive(y_true, y_pred, positive_label='good'):
    return np.sum((y_pred == positive_label) & (y_true != positive_label))

def false_negative(y_true, y_pred, positive_label='good'):
    return np.sum((y_pred != positive_label) & (y_true == positive_label))

def true_negative(y_true, y_pred, positive_label='good'):
    return np.sum((y_pred != positive_label) & (y_true != positive_label))

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, positive_label='good'):
    tp = true_positive(y_true, y_pred, positive_label)
    fp = false_positive(y_true, y_pred, positive_label)
    return tp / (tp + fp) if (tp + fp) else 0.0

def recall_score(y_true, y_pred, positive_label='good'):
    tp = true_positive(y_true, y_pred, positive_label)
    fn = false_negative(y_true, y_pred, positive_label)
    return tp / (tp + fn) if (tp + fn) else 0.0

def f1_score(y_true, y_pred, positive_label='good'):
    precision = precision_score(y_true, y_pred, positive_label)
    recall = recall_score(y_true, y_pred, positive_label)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

def classification_metrics(y_true, y_pred, positive_label='good'):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, positive_label),
        "recall": recall_score(y_true, y_pred, positive_label),
        "f1_score": f1_score(y_true, y_pred, positive_label)
    }


def confusion_matrix(y_true, y_pred, positive_label='good'):
    TP = true_positive(y_true, y_pred, positive_label)
    FP = false_positive(y_true, y_pred, positive_label)
    FN = false_negative(y_true, y_pred, positive_label)
    TN = true_negative(y_true, y_pred, positive_label)

    cm = pd.DataFrame(
        [[TP, FN],
         [FP, TN]],
        index=[f'Actual {positive_label}', f'Actual not-{positive_label}'],
        columns=[f'Predicted {positive_label}', f'Predicted not-{positive_label}']
    )
    return cm


import numpy as np

def roc_auc_score(y_true, scores, pos_label='good'):
    """
    Fixed ROC AUC calculation that:
    1. Handles edge cases (all positive/negative predictions)
    2. Guarantees correct threshold ordering
    3. Properly interpolates between points
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    
    if len(y_true) == 0 or len(scores) == 0:
        return np.array([0., 1.]), np.array([0., 1.]), 0.5
    
    desc_score_indices = np.argsort(scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    scores_sorted = scores[desc_score_indices]
    
    P = np.sum(y_true == pos_label)
    N = len(y_true) - P
    
    # Edge case: All predictions same
    if np.all(scores == scores[0]):
        pred_class = (scores[0] >= 0.5)  # Arbitrary threshold
        TP = np.sum((y_true == pos_label) & pred_class)
        FP = np.sum((y_true != pos_label) & pred_class)
        TPR = TP / P if P > 0 else 0.
        FPR = FP / N if N > 0 else 0.
        return np.array([0., FPR, 1.]), np.array([0., TPR, 1.]), max(TPR, 1-FPR)
    
    # Calculate ROC points
    tprs = [0.]
    fprs = [0.]
    
    for i in range(1, len(scores_sorted)):
        if scores_sorted[i] != scores_sorted[i-1]:
            thresh = scores_sorted[i-1]
            preds = scores >= thresh
            TP = np.sum((preds) & (y_true == pos_label))
            FP = np.sum((preds) & (y_true != pos_label))
            tprs.append(TP / P if P > 0 else 0.)
            fprs.append(FP / N if N > 0 else 0.)
    
    tprs.append(1.)
    fprs.append(1.)
    
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2
    
    return np.array(fprs), np.array(tprs), auc
