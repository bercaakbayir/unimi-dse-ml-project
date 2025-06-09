import numpy as np

def classification_metrics(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    metrics = {}

    total_correct = np.sum(y_true == y_pred)
    accuracy = total_correct / len(y_true)
    metrics['accuracy'] = accuracy

    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics['precision'] = np.mean(precisions)
    metrics['recall'] = np.mean(recalls)
    metrics['f1_score'] = np.mean(f1s)

    return metrics