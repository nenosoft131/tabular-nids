from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
# from catboost import CatBoostClassifier
import numpy as np
from utils import Utils

class supervised():
    def __init__(self, seed:int, classes, model_name:str=None, ):
        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.classes = classes
        self.model_dict = {'LR':LogisticRegression,
                           'NB':GaussianNB,
                           'SVM':SVC,
                           'MLP':MLPClassifier,
                           'RF':RandomForestClassifier,
                           'LGB':lgb.LGBMClassifier,
                           'XGB':xgb.XGBClassifier}
                        #    'CatB':CatBoostClassifier}

    def fit(self, X_train, y_train, **kwargs):
        if self.model_name == 'LR': 
            penalty = kwargs.get("penalty")
            C = kwargs.get("C")
            class_weight = kwargs.get("class_weight")
            solver = kwargs.get("solver")
            self.model = self.model_dict[self.model_name](random_state=self.seed, penalty=penalty, C=C, class_weight=class_weight, solver=solver)

        elif self.model_name == 'SVM':
            kernel = kwargs.get("kernel", 10)
            gamma = kwargs.get("gamma")
            coef0 = kwargs.get("coef0")
            tol =kwargs.get("tol")
            C = kwargs.get("C")
            self.model = self.model_dict[self.model_name](kernel=kernel, gamma=gamma, coef0=coef0, tol=tol, C=C,probability=True)
        
        elif self.model_name == 'MLP':
            act_fun = kwargs.get("act_fun")
            solver = kwargs.get("solver")
            batch_size = kwargs.get("batch_size")
            hidden_layer_sizes =kwargs.get("hidden_layer_sizes")
            alpha = kwargs.get("alpha")
            max_iter = kwargs.get("max_iter")
            self.model = self.model_dict[self.model_name](random_state=self.seed, activation=act_fun, batch_size=batch_size, solver=solver, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=max_iter)
        
        elif self.model_name == 'RF':
            for cls in self.classes:
                # X_cls = X_train[y_train == cls] 
                scaler = StandardScaler()  # Initialize the scaler
                X_scaled = scaler.fit_transform(X_train) 
                # print(f"Shape of X_scaled: {X_scaled.shape}")
                # print(f"Shape of y_cls: {y_cls.shape}")
                class_weight = kwargs.get("class_weight")
                n_estimators = kwargs.get("n_estimators")
                max_depth = kwargs.get("max_depth")
                min_samples_split =kwargs.get("min_samples_split")
                min_samples_leaf = kwargs.get("min_samples_leaf")
                max_features = kwargs.get("max_features")
                ccp_alpha = kwargs.get("ccp_alpha")
                self.model = self.model_dict[self.model_name](random_state=self.seed, class_weight=class_weight, n_estimators=n_estimators, max_depth=max_depth, 
                                                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                                          max_features=max_features, ccp_alpha=ccp_alpha)
        
        elif self.model_name == 'XGB':
            learning_rate = kwargs.get("learning_rate")
            max_depth = kwargs.get("max_depth")
            n_estimators = kwargs.get("n_estimators")
            reg_alpha = kwargs.get("reg_alpha")
            reg_lambda = kwargs.get("reg_lambda")
            self.model = self.model_dict[self.model_name](random_state=self.seed, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
        
        else:
            self.model = self.model_dict[self.model_name](random_state=self.seed)
            #
        # else:
            # self.model = self.model_dict[self.model_name](random_state=self.seed, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, num_leaves=num_leaves, subsample=subsample)
            # self.model = self.model_dict[self.model_name](random_state=self.seed, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
           
            # self.model = self.model_dict[self.model_name](random_state=self.seed, activation=act_fun, batch_size=batch_size, solver=solver, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=max_iter)
            # self.model = self.model_dict[self.model_name](random_state=self.seed, class_weight=class_weight, n_estimators=n_estimators, max_depth=max_depth, 
            #                                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            #                                               max_features=max_features, ccp_alpha=ccp_alpha)
            # self.model = self.model_dict[self.model_name](random_state=self.seed, penalty=penalty, C=C, class_weight=class_weight, solver=solver, max_iter=max_iter)
            
        # fitting
        self.model.fit(X_scaled, y_train)

        return self

    def predict_score(self, X_test, X_val):
        # score = self.model.predict_proba(X)[:, 1]
        val_probs = self.model.predict_proba(X_val)[:, 1]
        test_probs = self.model.predict_proba(X_test)[:, 1]
        return  test_probs, val_probs
    
    def predict_train_score(self, X_test):
        test_probs = self.model.predict_proba(X_test)[:, 1]
        return test_probs 
    
    def predict(self, X):
        predictions = []
        scores = []
        for x in X:
            sample_scores = {
                cls: self.model[cls].predict_train_score(self.scalers[cls].transform([x]))[0]
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