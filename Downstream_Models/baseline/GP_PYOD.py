from cuml.neighbors import NearestNeighbors
from eilof import EILOF 
from cuml.decomposition import PCA
import numpy as np
from cuml.svm import SVC as cumlSVC
from logger_config import get_logger

class GP_PYOD:
    def __init__(self, seed, model_name, device=None):
        self.seed = seed
        self.model_name = model_name
        self.model_dict = {'cuKNN':NearestNeighbors, 'cuLOF':EILOF, 'cuSVM': cumlSVC} # default value; will be overridden by config
        self.logger = get_logger(__name__)

    def fit(self, X_train, y_train=None, **kwargs):
        if self.model_name == 'cuKNN':
            n_neighbors = kwargs.get('n_neighbors', 5)
            algorithm = kwargs.get("algorithm", "auto")
            metric = kwargs.get("metric", "euclidean")
            self.model = self.model_dict[self.model_name](n_neighbors=n_neighbors, algorithm=algorithm, metric=metric).fit(X_train)
            # self.model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
            # self.model.fit(X_train)
        elif self.model_name == 'cuLOF':
            k = 20
            batch_size = 5000

            # First chunk initializes the model
            # model.fit(X_train_ref[:batch_size])
            # self.model = self.model_dict[self.model_name](k=k).fit(X_train[:batch_size])
            # self.model = self.model_dict[self.model_name](k=k)
            # first_batch = first_batch[~np.all(first_batch == 0, axis=1)]  # remove all-zero rows
            # self.model.fit(X_train[:batch_size])
            
            
            first_batch = X_train[:batch_size]
            first_batch = first_batch[~np.all(first_batch == 0, axis=1)]  # remove all-zero rows
            self.model = self.model_dict[self.model_name](k=k)
            self.model.fit(X_train[:batch_size])
            # for start in range(batch_size, len(X_train), batch_size):
            #     end = min(start + batch_size, len(X_train))
            #     batch = X_train[start:end]
            #     batch = batch[~np.all(batch == 0, axis=1)]

            #     if batch.shape[0] > 0:
            #         try:
            #             self.model.update(new_points=batch)  # ✅ call update on instance, not class
            #         except Exception as e:
            #             print(f"Update failed for batch {start}:{end} - {e}")
        elif self.model_name == 'cuSVM':
            n_neighbors = kwargs.get('n_neighbors', 5)
            self.model = self.model_dict[self.model_name](n_neighbors=n_neighbors, algorithm=algorithm, metric=metric).fit(X_train)
                
        return self

    def predict_score(self, X_test, X_val):
        # Compute distances to nearest neighbors
        if self.model_name == 'cuLOF':
            # get_logger(__name__).info("ENTER")
            # get_logger(__name__).info(f"Test shape: {X_test.shape}, Val shape: {X_val.shape if X_val is not None else 'None'}")
            # self.model_dict[self.model_name].update(X_val)
            # get_logger(__name__).info("ENTER 2")
            test_results = self.model.predict_labels(threshold=95)
            val_scores = self.model.predict_labels(threshold=95)
            return test_results, val_scores
        
            # return self.model_dict[self.model_name].predict(X_test), self.model_dict[self.model_name].predict(X_val) if X_val is not None else None
        else:
            distances_test, _ = self.model.kneighbors(X_test)
            test_scores = distances_test.mean(axis=1)
            val_distances, _ = self.model.kneighbors(X_val)
            val_scores = val_distances.mean(axis=1)
        return test_scores, val_scores

    def predict_train_score(self, x_train):
        if self.model_name == 'cuKNN':
            distances_train, _ = self.model.kneighbors(x_train)
            train_scores = distances_train.mean(axis=1)
        
        elif self.model_name == 'cuLOF':
            # get_logger(__name__).info("ENTER")
            # get_logger(__name__).info(f"Test shape: {X_test.shape}, Val shape: {X_val.shape if X_val is not None else 'None'}")
            # self.model_dict[self.model_name].update(X_val)
            # get_logger(__name__).info("ENTER 2")
            train_scores = self.model.predict_labels(threshold=95)
            print("PREDICTION DONE")
        return train_scores