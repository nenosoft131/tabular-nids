from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize, StandardScaler
import numpy as np
import pandas as pd


class OVR_OCSVM:
    def __init__(self, classes, nu=0.1, gamma='scale'):
        self.classes = classes
        self.nu = nu
        self.gamma = gamma
        self.models = {}
        self.scalers = {}

    def fit(self, X_train, y_train):
        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cls)
            model = OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)
            model.fit(X_scaled)
            self.models[cls] = model
            self.scalers[cls] = scaler

    def predict(self, X):
        predictions = []
        scores = []
        for x in X:
            sample_scores = {
                cls: self.models[cls].decision_function(self.scalers[cls].transform([x]))[0]
                for cls in self.classes
            }
            predictions.append(max(sample_scores, key=sample_scores.get))
            scores.append([sample_scores[c] for c in self.classes])
        return np.array(predictions), np.array(scores)

    def evaluate(self, X, y_true, stage='Test'):
        from sklearn.metrics import (
            classification_report, accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, roc_auc_score, average_precision_score
        )
        from sklearn.preprocessing import label_binarize
        import pandas as pd

        y_pred, y_scores = self.predict(X)

        # One-hot encoding for AUC/PR
        try:
            y_true_bin = label_binarize(y_true, classes=self.classes)
            aucroc = roc_auc_score(y_true_bin, y_scores, average='macro', multi_class='ovr')
            aucpr = average_precision_score(y_true_bin, y_scores, average='macro')
        except:
            aucroc = None
            aucpr = None

        # Per-class metrics
        class_report = classification_report(y_true, y_pred, labels=self.classes, output_dict=True, zero_division=0)
        class_report_df = pd.DataFrame(class_report).transpose()

        print(f"\n🔍 Per-Class Metrics ({stage}):\n")
        print(class_report_df.loc[[str(cls) for cls in self.classes]])

        # Lowercase stage name for keys
        stage_key = stage.lower()

        results = {
            f'{stage_key}_aucroc': aucroc,
            f'{stage_key}_aucpr': aucpr,
            f'{stage_key}_accuracy': accuracy_score(y_true, y_pred),
            f'{stage_key}_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            f'{stage_key}_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            f'{stage_key}_f1': f1_score(y_true, y_pred, average='weighted'),
            f'{stage_key}_f1_macro': f1_score(y_true, y_pred, average='macro'),
            f'{stage_key}_f1_micro': f1_score(y_true, y_pred, average='micro'),
            f'{stage_key}_confusion_matrix': confusion_matrix(y_true, y_pred, labels=self.classes),
            f'{stage_key}_per_class_metrics': class_report_df.loc[[str(cls) for cls in self.classes]]
        }

        return results


