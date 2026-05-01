import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

class F1ThresholdOptimizer:
    def __init__(self):
        self.best_threshold = None
        self.best_f1 = None

    def fit(self, y_true, y_score, positive_label=1):
        # Flip scores so higher = more anomalous
        scores = -np.asarray(y_score)
        y_true = np.asarray(y_true)

        precision, recall, thresholds = precision_recall_curve(y_true, scores, pos_label=positive_label)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        best_idx = np.argmax(f1_scores)
        self.best_threshold = thresholds[best_idx]
        self.best_f1 = f1_scores[best_idx]
        return self.best_threshold, self.best_f1

    def predict(self, y_score):
        if self.best_threshold is None:
            raise ValueError("You must call fit() before predict().")
        return (-np.asarray(y_score) >= self.best_threshold).astype(int)
