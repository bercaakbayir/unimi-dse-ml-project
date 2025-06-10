import numpy as np

def true_positive(y_true, y_pred, positive_label='good'):
    return np.sum((y_pred == positive_label) & (y_true == positive_label))

def false_positive(y_true, y_pred, positive_label='good'):
    return np.sum((y_pred == positive_label) & (y_true != positive_label))

def false_negative(y_true, y_pred, positive_label='good'):
    return np.sum((y_pred != positive_label) & (y_true == positive_label))

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
