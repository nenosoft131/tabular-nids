from cuml.svm import SVC as cuSVC
from logger_config import get_logger

class SVM_GPU:
    def __init__(self, seed=42, model_name='SVM', device=None):
        self.seed = seed
        self.model_name = model_name
        self.logger = get_logger(__name__)
        self.model_dict = {'SVM': cuSVC}
        self.model = None

    def fit(self, X_train, y_train, **kwargs):
        if self.model_name == 'SVM':
            kernel = kwargs.get('kernel', 'rbf')
            gamma = kwargs.get('gamma', 'scale')
            C = kwargs.get('C', 1.0)
            tol = kwargs.get('tol', 1e-3)
            coef0 = kwargs.get('coef0', 0.0)


            self.model = self.model_dict[self.model_name](
                kernel=kernel,
                gamma=gamma,
                C=C,
                tol=tol,
                coef0=coef0,
                probability=False  # Set to True only if you need probabilities (slower)
            )
            self.model.fit(X_train, y_train)
        return self

    def predict_score(self, X_test, X_val=None):
        if self.model_name == 'SVM':
            test_preds = self.model.predict(X_test)
            val_preds = self.model.predict(X_val) if X_val is not None else None
            return test_preds, val_preds
        return None, None

    def predict_train_score(self, X_train):
        if self.model_name == 'SVM':
            return self.model.predict(X_train)
        return None
