import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import gc
# from keras import backend as K
import tensorflow as tf
K = tf.keras.backend
from logger_config import get_logger
from data_generator import DataGenerator
# from utils import Utils
from gymconf import gym
import torch.nn as nn
import torch
# most of the codes are from the ADBench (NeurIPS 2022)
# notice that the compared SOTA models should use the same training settings in ADGym
class RunPipeline():
    def __init__(self,
                 suffix: str=None,
                 mode: str='rla',
                 parallel: str=None,
                 generate_duplicates: bool=False,
                 n_samples_lower_bound: int=1000,
                 n_samples_upper_bound: int=3000):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_lower_bound: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_lower_bound will be dropped
        :param n_samples_upper_bound: threshold for downsampling input samples, considering the computational cost
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger(__name__)
        # utils function
        # self.utils = Utils()
        self.mode = mode
        self.parallel = parallel
        self.score_train = None

        # global parameters
        self.generate_duplicates = generate_duplicates
        self.n_samples_lower_bound = n_samples_lower_bound
        self.n_samples_upper_bound = n_samples_upper_bound


        # the suffix of all saved files
        self.suffix = suffix + '-' + self.parallel

        if not os.path.exists('result'):
            os.makedirs('result')

        # data generator instantiation
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_lower_bound=self.n_samples_lower_bound,
                                            n_samples_upper_bound=self.n_samples_upper_bound)

    
        # seed list
        self.seed_list = list(range(1, 2))

        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'Bert':
            from baseline.BOD import BOD

            # from pyod
            # for _ in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
            #           'LOF', 'LSCP', 'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL']:
            for _ in ['Bert']:
                self.model_dict[_] = BOD

    # dataset filter for delelting those datasets that do not satisfy the experimental requirement
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = [os.path.splitext(f)[0] for f in os.listdir('datasets')
                        if os.path.splitext(f)[1].lower() == '.csv']
        print(f"Original dataset list: {dataset_list_org}")
        dataset_list, dataset_size = [], []
        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset

                try:
                    data = self.data_generator.generator(at_least_one_labeled=True)
                except:
                    add = False
                    # print("Error in data generation for dataset:")
                    pass
                    continue
            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['X_train']))
        # sort datasets by their sample size
        dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]
        return dataset_list

    # model fitting function
    def model_fit(self,model_name,params):
        self.score_train = None
        gym_search_space = gym(self=None, mode=model_name, max_combinations=1)
        for config in gym_search_space:
            try:
                self.model_name = model_name
                self.clf = self.model_dict[self.model_name]
                 # Model Initialization
                if 'tensorflow' in str(type(self.clf)).lower() or 'keras' in str(type(self.clf)).lower():
                    strategy = tf.distribute.MirroredStrategy()
                    with strategy.scope():
                        self.clf = self.clf(seed=self.seed, model_name=self.model_name)

                elif 'torch' in str(type(self.clf)).lower() or 'sklearn' not in str(type(self.clf)).lower():
                    # PyTorch model
                    self.clf = self.clf(seed=self.seed, model_name=self.model_name)
                    # Move model to GPU
                    if isinstance(self.clf, nn.Module):
                        self.clf.to(self.device)

                else:
                    # Sklearn or other CPU-based models
                    self.clf = self.clf(seed=self.seed, model_name=self.model_name)

            except Exception as error:
                print(f'Error in model initialization. Model: {self.model_name}, Error: {error}')
                continue

            try:
                # fitting
                start_time = time.time()
                train_ds = self.data['X_train']
                val_ds = self.data['X_val']
                
                anomaly_rate = 0.038
                get_logger(__name__).info(f"Final Anomaly rate: {anomaly_rate}")

                
                # Convert to torch tensors and move to GPU if needed
                if 'torch' in str(type(self.clf)).lower() or isinstance(self.clf, nn.Module):
                    if not isinstance(X_train, torch.Tensor):
                        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                    if not isinstance(y_train, torch.Tensor) and len(y_train) > 0:
                        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)

                if model_name == 'Bert':
                    self.clf = self.clf.fit(train_ds, val_ds,
                                            learning_rate=config['learning_rate'], per_device_train_batch_size=config['per_device_train_batch_size'],
                                            per_device_eval_batch_size=config['per_device_eval_batch_size'], num_train_epochs=config['num_train_epochs'] )
                
                end_time = time.time(); time_fit = end_time - start_time

                # predicting score (inference)
                start_time = time.time()
                
                
                test_ds = self.data['X_test']

                if isinstance(self.clf, nn.Module):
                    if not isinstance(X_test, torch.Tensor):
                        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                    if X_val is not None and not isinstance(X_val, torch.Tensor):
                        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)

                
                self.score_train = self.clf.predict_score(test_ds)
                print(f"Model: {self.model_name}, Score Train: {self.score_train}")
                # get_logger(__name__).info("Calculating Score")
                # if self.score_train is None:
                #     get_logger(__name__).info("Score train calcualted")
                #     self.score_train = self.clf.predict_train_score(self.data['X_train'])
                # else:
                #     get_logger(__name__).info("No score train needed")
                # score_test , score_val = self.clf.predict_score(self.data['X_test'], self.data['X_val'])
                    
                # get_logger(__name__).info(f"score train{self.score_train}")
                
                exit()
                # score_train, score_test , score_val = self.clf.predict_score(self.data['X_train'],self.data['X_test'],self.data['X_val'])
                end_time = time.time(); time_inference = end_time - start_time
                # performance
                # result = self.utils.metric(self.score_train, anomaly_rate, y_true=self.data['y_test'], y_score=score_test, val_true=self.data['y_val'] , val_score=score_val, pos_label=1)
                # K.clear_session()
                # print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")
                # get_logger(__name__).info(config['epochs'])
                # get_logger(__name__).info(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")
                # del self.clf
                # gc.collect()
                # Cleanup memory
                if 'tensorflow' in str(type(self.clf)).lower():
                    K.clear_session()
                elif isinstance(self.clf, nn.Module):
                    del self.clf
                gc.collect()
                
                
                df_AUCROC = pd.DataFrame({'model_name': [model_name], 'params': [params], 'config': [config],
                'AUCROC-Val':        [result['val_aucroc']],
                'AUCPR-Val':         [result['val_aucpr']],
                'Accuracy-Val':      [result['val_accuracy']],
                'Precision-Val':     [result['val_precision']],
                'Recall-Val':        [result['val_recall']],
                'F1-Val':            [result['val_f1']],
                'F1-Macro-Val':      [result['val_f1_macro']],
                'F1-Micro-Val':      [result['val_f1_micro']],
                'Confusion-Matrix-Val': [result['val_confusion_matrix']],

                'AUCROC-Test':       [result['test_aucroc']],
                'AUCPR-Test':        [result['test_aucpr']],
                'Accuracy-Test':     [result['test_accuracy']],
                'Precision-Test':    [result['test_precision']],
                'Recall-Test':       [result['test_recall']],
                'F1-Test':           [result['test_f1']],
                'F1-Macro-Test':     [result['test_f1_macro']],
                'F1-Micro-Test':     [result['test_f1_micro']],
                'Confusion-Matrix-Test': [result['test_confusion_matrix']],
                'TIME_FIT' :[time_fit], 'TIME_INFER':[time_inference]})
                
                
                
                # df_AUCROC = pd.DataFrame({'model_name': [model_name], 'params': [params], 'config': [config], 'AUCROC': [result['aucroc']], 'AUCPR': [result['aucpr']], 'TIME_FIT' :[time_fit], 'TIME_INFER':[time_inference]})
                
                # df_AUCPR = pd.DataFrame({'model_name': [model_name], 'params': [params], 'config': [config], 'AUCPR': [result['aucpr']]})

                # df_AUCROC = pd.concat([df_AUCROC, new_row], ignore_index=True)
                
                # df_AUCPR[model_name].iloc[i] = result['aucpr']
                # df_time_fit[model_name].iloc[i+ count] = time_fit
                # df_time_inference[model_name].iloc[i] = time_inference
                
                # df_AUCROC.loc[i, model_name] = result['aucroc']
                # df_AUCPR.loc[i, model_name] = result['aucpr']
                # df_time_fit.loc[i, model_name] = time_fit
                # df_time_inference.loc[i, model_name] = time_inference
                file_path = os.path.join(os.getcwd(), 'result', 'AUCROC-' + self.suffix + '.csv')
                # file_path_pr = os.path.join(os.getcwd(), 'result', 'AUCPR-' + self.suffix + '.csv')

                df_AUCROC.to_csv(file_path, mode='a', index=True, header=not os.path.exists(file_path))
                
                

            except Exception as error:
                print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
                time_fit, time_inference = None, None
                result = {'aucroc': np.nan, 'aucpr': np.nan}
                pass
        
        return time_fit, time_inference, result

    # run the experiment
    def run(self):
        #  filteting dataset that does not meet the experimental requirements
        dataset_list = self.dataset_filter()
        print(f'Filtered dataset list: {dataset_list}')
        # experimental parameters
        experiment_params = list(product(dataset_list, self.seed_list))

        print(f'{len(dataset_list)} datasets, {len(self.model_dict.keys())} models')
        print(f'Experiment parameters: {len(experiment_params)}')
        
        
        for i, params in tqdm(enumerate(experiment_params)):
            dataset, self.seed = params

    
            print(f'Current experiment parameters: {params}')

            # generate data
            self.data_generator.seed = self.seed
            self.data_generator.dataset = dataset

            try:
                self.data = self.data_generator.generator(at_least_one_labeled=True)

            except Exception as error:
                print(f'Error when generating data: {error}')
                pass
                continue
            for model_name in tqdm(self.model_dict.keys()):
                time_fit, time_inference, result = self.model_fit(model_name, params)




get_logger(__name__).info("BERT Started")
print("BERT Started......")
# run the above pipeline for reproducing the results in the paper
pipeline = RunPipeline(suffix='SOTA', parallel='Bert', mode='nla')
pipeline.run()