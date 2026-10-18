import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def binary_metrics(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid probability')
    predicted = (probability >= 0.5).astype(int)
    return {'accuracy': float(accuracy_score(y_true, predicted)),
            'macro_f1': float(f1_score(y_true, predicted, average='macro')),
            'roc_auc': float(roc_auc_score(y_true, probability)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, predicted))}
