from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import lightgbm as lgb
import xgboost as xgb

import numpy as np

class supervised:
    def __init__(self, seed: int, classes=None, model_name: str = None):
        self.seed = seed
        self.model_name = model_name
        self.classes = np.array(classes) if classes is not None else None
        self.model = None

        self.model_dict = {
            'LR': LogisticRegression,
            'NB': GaussianNB,
            'SVM': SVC,
            'MLP': MLPClassifier,
            'RF': RandomForestClassifier,
            'LGB': lgb.LGBMClassifier,
            'XGB': xgb.XGBClassifier,
        }

    # ----------- helpers -----------
    def _needs_scaler(self, name: str) -> bool:
        # Scale features for these models
        return name in {'LR', 'SVM', 'MLP'}

    def _ensure_classes(self, y):
        if self.classes is None:
            self.classes = np.unique(y)
        else:
            # Keep provided order but ensure coverage
            missing = set(np.unique(y)) - set(self.classes)
            if missing:
                self.classes = np.unique(y)  # fall back to y order to stay consistent

    # ----------- fit -----------
    def fit(self, X_train, y_train, **kwargs):
        self._ensure_classes(y_train)
        n_classes = len(self.classes)

        name = self.model_name
        if name is None or name not in self.model_dict:
            raise ValueError(f"Unknown or missing model_name: {name}")

        # Common kwargs with safe defaults
        class_weight = kwargs.get("class_weight", None)  # None | 'balanced' | dict

        # Optionally auto-compute dict weights
        if class_weight == "balanced_dict":
            weights = compute_class_weight("balanced", classes=self.classes, y=y_train)
            class_weight = {c: float(w) for c, w in zip(self.classes, weights)}

        # Build the estimator per family
        if name == 'LR':
            # Use multinomial when possible
            penalty = kwargs.get("penalty", 'l2')
            C = kwargs.get("C", 1.0)
            solver = kwargs.get("solver", 'lbfgs')  # lbfgs supports multinomial
            max_iter = kwargs.get("max_iter", 1000)

            base = LogisticRegression(
                random_state=self.seed, penalty=penalty, C=C,
                class_weight=class_weight, solver=solver, max_iter=max_iter,
                multi_class='auto'
            )

        elif name == 'SVM':
            kernel = kwargs.get("kernel", 'rbf')
            gamma = kwargs.get("gamma", 'scale')
            coef0 = kwargs.get("coef0", 0.0)
            tol = kwargs.get("tol", 1e-3)
            C = kwargs.get("C", 1.0)
            # probability=True enables predict_proba (uses Platt scaling)
            base = SVC(kernel=kernel, gamma=gamma, coef0=coef0, tol=tol, C=C,
                       probability=True, class_weight=class_weight,
                       random_state=self.seed)

        elif name == 'MLP':
            act_fun = kwargs.get("act_fun", 'relu')
            solver = kwargs.get("solver", 'adam')
            batch_size = kwargs.get("batch_size", 'auto')
            hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (128, 64))
            alpha = kwargs.get("alpha", 1e-4)
            max_iter = kwargs.get("max_iter", 300)
            base = MLPClassifier(
                random_state=self.seed, activation=act_fun, batch_size=batch_size,
                solver=solver, hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha, max_iter=max_iter
            )

        elif name == 'NB':
            base = GaussianNB()  # No scaling required; continuous features assumed.

        elif name == 'RF':
            n_estimators = kwargs.get("n_estimators", 300)
            max_depth = kwargs.get("max_depth", None)
            min_samples_split = kwargs.get("min_samples_split", 2)
            min_samples_leaf = kwargs.get("min_samples_leaf", 1)
            max_features = kwargs.get("max_features", "sqrt")
            ccp_alpha = kwargs.get("ccp_alpha", 0.0)

            base = RandomForestClassifier(
                random_state=self.seed, class_weight=class_weight,
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                max_features=max_features, ccp_alpha=ccp_alpha, n_jobs=-1
            )

        elif name == 'LGB':
            learning_rate = kwargs.get("learning_rate", 0.05)
            n_estimators = kwargs.get("n_estimators", 500)
            num_leaves = kwargs.get("num_leaves", 31)
            subsample = kwargs.get("subsample", 1.0)
            colsample_bytree = kwargs.get("colsample_bytree", 1.0)

            base = lgb.LGBMClassifier(
                random_state=self.seed,
                objective='multiclass',
                num_class=n_classes,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                class_weight=class_weight
            )

        elif name == 'XGB':
            learning_rate = kwargs.get("learning_rate", 0.05)
            max_depth = kwargs.get("max_depth", 6)
            n_estimators = kwargs.get("n_estimators", 500)
            reg_alpha = kwargs.get("reg_alpha", 0.0)
            reg_lambda = kwargs.get("reg_lambda", 1.0)
            subsample = kwargs.get("subsample", 1.0)
            colsample_bytree = kwargs.get("colsample_bytree", 1.0)

            base = xgb.XGBClassifier(
                random_state=self.seed,
                objective='multi:softprob',
                num_class=n_classes,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                tree_method='hist',
                n_jobs=-1,
                eval_metric='mlogloss'
            )

        else:
            # Fallback (shouldn't happen due to earlier check)
            base = self.model_dict[name](random_state=self.seed)

        # Wrap with scaler when needed
        if self._needs_scaler(name):
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", base)
            ])
        else:
            self.model = base

        # Fit once on all data (multiclass)
        self.model.fit(X_train, y_train)
        return self

    # ----------- inference -----------
    def predict_proba(self, X):
        """Return (n_samples, n_classes) in the order of self.classes_."""
        proba = self.model.predict_proba(X)
        # Align columns to self.classes
        # For pipelines, clf is the last step
        clf = self.model.named_steps["clf"] if isinstance(self.model, Pipeline) else self.model
        model_classes = np.array(getattr(clf, "classes_", self.classes))
        if not np.array_equal(model_classes, self.classes):
            # Reorder columns to match self.classes
            idx = [np.where(model_classes == c)[0][0] for c in self.classes]
            proba = proba[:, idx]
        return proba

    def predict(self, X):
        """Return class predictions aligned with self.classes order."""
        return self.model.predict(X)

    # ----------- convenience (kept similar names) -----------
    def predict_score(self, X_test, X_val):
        """Return probability matrices for all classes."""
        val_probs = self.predict_proba(X_val)  # shape (n_val, K)
        test_probs = self.predict_proba(X_test)  # shape (n_test, K)
        return test_probs, val_probs

    def predict_train_score(self, X):
        """Return probability matrix for provided X."""
        return self.predict_proba(X)

    # ----------- evaluation -----------
    def evaluate(self, X, y_true, stage='Test'):
        from sklearn.metrics import (
            classification_report, accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, roc_auc_score, average_precision_score
        )
        from sklearn.preprocessing import label_binarize
        import pandas as pd

        y_pred = self.predict(X)
        y_scores = self.predict_proba(X)  # (n_samples, K)

        # AUC/PR (macro) using OvR if multiclass
        aucroc = None
        aucpr = None
        try:
            if len(self.classes) > 2:
                y_true_bin = label_binarize(y_true, classes=self.classes)
                aucroc = roc_auc_score(y_true_bin, y_scores, average='macro', multi_class='ovr')
                aucpr = average_precision_score(y_true_bin, y_scores, average='macro')
            else:
                # Binary: take prob for positive class (classes[1])
                pos_idx = 1 if self.classes[0] == 0 and self.classes[1] == 1 else 1
                aucroc = roc_auc_score(y_true, y_scores[:, pos_idx])
                aucpr = average_precision_score(y_true, y_scores[:, pos_idx])
        except Exception:
            pass

        # Per-class report
        class_report = classification_report(
            y_true, y_pred, labels=self.classes, output_dict=True, zero_division=0
        )
        class_report_df = pd.DataFrame(class_report).transpose()

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
            f'{stage_key}_per_class_metrics': class_report_df.loc[[str(cls) for cls in self.classes.astype(str)]]
        }
        return results
