import itertools

def gym(self, mode='small', max_combinations=10):
    if mode == 'large':
        gyms = {}
        # gyms['augmentation'] = [None, 'Oversampling', 'SMOTE', 'Mixup', 'GAN']
        # gyms['preprocess'] = ['minmax', 'normalize']
        # gyms['network_architecture'] = ['MLP', 'AE', 'ResNet', 'FTT']
        # gyms['hidden_size_list'] = [[20], [100, 20], [100, 50, 20]]
        # gyms['act_fun'] = ["Tanh", "ReLU", "Softmax"]
        # gyms['act_fun'] = ['relu', 'tanh']
        # gyms['network_depth'] = ['1', '2', '4']
        # gyms['dropout'] = [0.0, 0.1, 0.2]
        # gyms['network_initialization'] = ['default', 'pretrained', 'xavier_normal', 'kaiming_normal']
        # gyms['loss_name'] = ['bce', 'focal', 'minus', 'inverse', 'hinge', 'deviation']
        # gyms['optimizer_name'] = ['SGD', 'Adam', 'RMSprop']
        # gyms['batch_resample'] = [True, False]
        # # gyms['epochs'] = [50]
        # gyms['batch_size'] = [64, 256]
        # gyms['max_iter'] = [500, 1000, 2000]
        # gyms['nb_batch'] = [20, 30]
        # gyms['lr'] = [1e-5, 1e-4, 1e-3, 1e-2]
        # # gyms['weight_decay'] = [0.0, 1e-5, 1e-4, 1e-3]
        # gyms['n_estimators'] = [10, 100, 200]
        # gyms['max_samples'] = [50, 100]
        # gyms['max_features'] = [0.5, 0.7, 1.0]
        gyms['contamination'] = [0.1, 0.25, 0.5]
        # gyms['n_neighbors'] = [5, 10, 20, 25]         # Various values for the number of neighbors
        # gyms['method'] = ['largest', 'mean', 'median']
        gyms['svd_solver'] = ['auto', 'full', 'arpack', 'randomized']
        gyms['n_components'] =[5, 10, 15]
        # gyms['nu'] = [0.1, 0.5, 0.9]
        # gyms['degree'] = [2, 3, 5]
        # gyms['gamma'] = ["scale", "auto"]
        # gyms['coef0'] = [0.0, 0.5, 1.0, 2.0]
        # gyms['tol'] = [1e-4, 1e-3, 1e-1]
        # gyms['solver'] = ['lbfgs', 'sgd', 'adam']  
        # gyms['hidden_layer_sizes'] =[[20], [100, 20], [100, 50, 20]]  # Sizes of hidden layers
        # gyms['alpha'] = [0.0001, 0.01, 0.1]
        
        
        # gyms['n_clusters'] = [3, 5, 8, 10]              # Various values for the number of clusters
        # gyms['contamination'] = [0.05, 0.1, 0.2, 0.3]   # Contamination values within the (0, 0.5) range
        # # gyms['n_random_cuts'] = [20, 50, 100]
        # gyms['n_bins'] = [5, 10, 15, 20, "auto"]          # Number of bins, with "auto" for automatic selection
        # gyms['alpha'] = [0.01, 0.05, 0.1, 0.5, 0.9]       # Regularizer values within the (0, 1) range
        # gyms['tol'] = [0.1, 0.25, 0.5, 0.75, 0.9]         
        # gyms['clustering_estimator'] = [None, "KMeans", "DBSCAN"]  # Common estimators
        # gyms['alpha'] = [0.6, 0.75, 0.9]                # Values within the (0.5, 1) range
        # gyms['beta'] = [1.5, 3, 5, 7] 
        
        # gyms['penalty'] = ['l2', 'none']        # Types of penalty norms
        # # gyms['dual'] = [True, False]                                # Dual or primal formulation
        # gyms['tol'] = [1e-4, 1e-3, 1e-2, 1e-1]                      # Various tolerance values
        # gyms['C'] = [0.1, 0.5, 1.0, 10.0]                           # Regularization strengths
        # # gyms['fit_intercept'] = [True, False]                       # Include intercept or not
        # # gyms['intercept_scaling'] = [0.5, 1.0, 2.0, 5.0]            # Intercept scaling values for 'liblinear' solver
        # gyms['class_weight'] = [None, 'balanced']                   # Class weights
        # # gyms['random_state'] = [None, 0, 42, 100]                   # Random states for reproducibility
        # gyms['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']  # Optimization algorithms
        # gyms['max_iter'] = [1000, 1500, 5000]                      # Number of iterations for convergence

        # gyms['multi_class'] = ['auto', 'ovr', 'multinomial']        # Classification strategies
        # gyms['verbose'] = [0, 1, 2]                                 # Verbosity levels
        # gyms['warm_start'] = [True, False]                          # Option to reuse previous solution
        # gyms['n_jobs'] = [None, 1, -1]                              # Number of CPU cores
        # gyms['l1_ratio'] = [None, 0.25, 0.5, 0.75, 1.0]  
        
        # gyms['kernel'] = ["linear", "poly", "rbf", "sigmoid"]                      # Number of trees in the forest
        # gyms['max_depth'] = [None, 10, 50]                      # Maximum depth of the tree
        # gyms['min_samples_split'] = [2, 10]           # Minimum samples required to split a node
        # gyms['min_samples_leaf'] = [1, 10]           # Minimum samples required at a leaf node
        # gyms['max_features'] = ['auto', 'log2', None] # Number of features to consider for best split
        # gyms['ccp_alpha'] = [0.0, 0.01, 0.1]  
        
        # gyms['learning_rate'] = [0.01, 0.1, 0.2]                  # Impact of each tree on the outcome
        # gyms['max_depth'] = [3, 5, 7]                             # Maximum depth of each tree
        # gyms['n_estimators'] = [100, 500, 1000]                   # Number of trees (boosting rounds)
        # gyms['num_leaves'] = [31, 63, 127]
        # gyms['subsample'] = [0.5, 0.8, 1.0]
        # gyms['reg_alpha'] = [0, 0.1, 0.5]                         # L1 regularization (sparsity control)
        
        # gyms['reg_lambda'] = [0.1, 0.5, 1]     
    
    elif mode == 'LR':
        gyms = {}
        gyms['penalty'] = ['l2', 'none']        # Types of penalty norms
        gyms['C'] = [0.1, 0.5, 1.0, 10.0] 
        gyms['class_weight'] = [None, 'balanced']   
        gyms['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] 
    
    elif mode == 'SVM':
        gyms = {}  
        gyms['gamma'] = ["scale", "auto"]
        gyms['coef0'] = [0.0, 0.5, 1.0, 2.0]
        gyms['tol'] = [1e-4, 1e-3, 1e-1]
        gyms['C'] = [0.1, 0.5, 1.0, 10.0]
        gyms['kernel'] = ["linear", "poly", "rbf", "sigmoid"] 
    
    elif mode == 'MLP':
        gyms = {}  
        gyms['act_fun'] = ['relu', 'tanh']
        gyms['batch_size'] = [64, 256]
        gyms['max_iter'] = [500, 1000, 2000]
        gyms['solver'] = ['lbfgs', 'sgd', 'adam']
        gyms['hidden_layer_sizes'] =[[20], [100, 20], [100, 50, 20]]  # Sizes of hidden layers
        gyms['alpha'] = [0.0001, 0.01, 0.1]
    
    elif mode == 'RF':
        gyms = {}  
        gyms['n_estimators'] = [10, 100, 200]
        gyms['class_weight'] = [None, 'balanced'] 
        gyms['max_depth'] = [None, 10, 50]  
        gyms['min_samples_split'] = [2, 10]           # Minimum samples required to split a node
        gyms['min_samples_leaf'] = [1, 10]           # Minimum samples required at a leaf node
        gyms['max_features'] = ['auto', 'log2', None] # Number of features to consider for best split
        gyms['ccp_alpha'] = [0.0, 0.01, 0.1] 
    
    elif mode == 'XGB':
        gyms = {} 
        gyms['learning_rate'] = [0.01, 0.1, 0.2]                  # Impact of each tree on the outcome
        gyms['max_depth'] = [3, 5, 7]                             # Maximum depth of each tree
        gyms['n_estimators'] = [100, 500, 1000] 
        gyms['reg_alpha'] = [0, 0.1, 0.5]                         # L1 regularization (sparsity control)
        gyms['reg_lambda'] = [0.1, 0.5, 1] 
        
    elif mode == 'PCA':
        gyms = {}
        gyms['contamination'] = [0.1, 0.25, 0.5]
        gyms['svd_solver'] = ['auto', 'full', 'arpack', 'randomized']
        gyms['n_components'] =[5, 10, 15]
        
    elif mode == 'IForest':
        gyms = {} 
        gyms['n_estimators'] = [10, 100, 200]
        gyms['max_samples'] = [50, 100]
        gyms['max_features'] = [0.5, 0.7, 1.0]
        gyms['contamination'] = [0.1, 0.25, 0.5]
    
    elif mode == 'OCSVM':
        gyms = {}  
        gyms['nu'] = [0.1, 0.5, 0.9]
        gyms['degree'] = [2, 3, 5]
        gyms['gamma'] = ["scale", "auto"]
        gyms['coef0'] = [0.0, 0.5, 1.0, 2.0]
        gyms['tol'] = [1e-4, 1e-3, 1e-1]
        
    elif mode == 'cuKNN':
        gyms = {}
        gyms['n_neighbors'] = [3, 5, 10] 
        gyms['algorithm'] = ['auto', 'brute']
        gyms['metric'] = ['euclidean', 'manhattan', 'minkowski', 'cosine']
        
    elif mode == 'LOF':
       gyms = {}
       gyms['n_neighbors'] = [5, 10, 20] 
       gyms['algorithm'] = ['auto', 'ball_tree', 'kd_tree']
       gyms['contamination'] = [0.5]
    #    gyms['contamination'] = [0.1, 0.25, 0.5]
       
    elif mode == 'cuLOF':
       gyms = {}
       gyms['k'] = [5, 10, 20] 
    #    gyms['algorithm'] = ['auto', 'ball_tree', 'kd_tree']
    #    gyms['contamination'] = [0.1, 0.25, 0.5]
        
        
    elif mode == 'RF':
        gyms = {}
        gyms['class_weight'] = [None, 'balanced'] 
        gyms['max_depth'] = [None, 10, 50]                      # Maximum depth of the tree
        gyms['min_samples_split'] = [2, 10]           # Minimum samples required to split a node
        gyms['min_samples_leaf'] = [1, 10]           # Minimum samples required at a leaf node
        gyms['max_features'] = ['auto', 'log2', None] # Number of features to consider for best split
        gyms['ccp_alpha'] = [0.0, 0.01, 0.1]  
    
    elif mode == 'AutoEncoder':
        gyms = {}
        gyms['batch_size'] = [32, 64, 128]
        gyms['hidden_neurons'] =[[20], [100, 20], [100, 50, 20]]
        gyms['weight_decay'] = [1e-5, 1e-4, 1e-3]
        gyms['hidden_activation'] = ['relu', 'tanh', 'leaky_relu']
    
    # elif mode == 'LR':
    
    # elif mode == 'XGB':
    
    elif mode == 'MLP':
        gyms = {}
        gyms['act_fun'] = ['relu', 'tanh']
        gyms['batch_size'] = [64, 256]
        gyms['max_iter'] = [500, 1000, 2000]
        gyms['solver'] = ['lbfgs', 'sgd', 'adam']  
        gyms['hidden_layer_sizes'] =[[20], [100, 20], [100, 50, 20]]  # Sizes of hidden layers
    
    # elif mode == 'SVM':
    
    elif mode == 'small':
        gyms = {}
        gyms['augmentation'] = [None, 'Oversampling', 'SMOTE', 'Mixup', 'GAN']
        gyms['preprocess'] = ['minmax']
        gyms['network_architecture'] = ['MLP', 'AE', 'ResNet', 'FTT']
        gyms['hidden_size_list'] = [[100, 20]]
        gyms['act_fun'] = ['Tanh', 'ReLU', 'LeakyReLU']
        gyms['dropout'] = [0.0]
        gyms['network_initialization'] = ['default']
        gyms['loss_name'] = ['bce', 'focal']
        gyms['optimizer_name'] = ['SGD', 'Adam']
        gyms['batch_resample'] = [True, False]
        gyms['epochs'] = [20,50]
        gyms['batch_size'] = [256]
        gyms['lr'] = [1e-2, 1e-3]
        gyms['weight_decay'] = [1e-2]

    else:
        raise NotImplementedError

    # Generate limited combinations using itertools.product
    param_keys = list(gyms.keys())
    param_values = list(gyms.values())
    all_combinations = list(itertools.product(*param_values))

    if max_combinations and len(all_combinations) > max_combinations:
        all_combinations = all_combinations[:max_combinations]

    limited_combinations = [dict(zip(param_keys, combination)) for combination in all_combinations]

    return limited_combinations
