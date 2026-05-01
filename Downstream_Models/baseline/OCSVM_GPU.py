from thundersvm import OneClassSVC
import numpy as np
from logger_config import get_logger

class OCSVM_GPU:
    def __init__(self, seed=42, model_name='OCSVM', device=None):
        self.seed = seed
        self.model_name = model_name
        self.logger = get_logger(__name__)
        self.model_dict = {'OCSVM': OneClassSVC}
        self.model = None

    def fit(self, X_train, y_train=None, **kwargs):
        if self.model_name == 'OCSVM':
            kernel = kwargs.get('kernel', 'rbf')
            gamma = kwargs.get('gamma', 'scale')
            nu = kwargs.get('nu', 0.1)

            self.logger.info(f"Training OneClassSVC with kernel={kernel}, gamma={gamma}, nu={nu}")
            self.model = self.model_dict[self.model_name](kernel=kernel, gamma=gamma, nu=nu)
            self.model.fit(X_train)
        return self

    def predict_score(self, X_test, X_val):
        if self.model_name == 'OCSVM':
            test_results = self.model.predict(X_test)  # returns 1 for inlier, -1 for outlier
            val_results = self.model.predict(X_val)
            return test_results, val_results
        return None, None

    def predict_train_score(self, X_train):
        if self.model_name == 'OCSVM':
            train_results = self.model.predict(X_train)
            return train_results
        return None
