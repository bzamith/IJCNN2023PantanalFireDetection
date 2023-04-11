"""
Config: Classification Settings
-----------------------------------
"""

NB_EPOCHS = 200
ALGORITHM = [
    "XGBoost",
    "Naive Bayes",
    "Random Forest",
    "Multi Layer Perceptron",
    "Support Vector Machine",
    "K-Nearest Neighbours",
    "Decision Tree",
    "LightGBM",
    "CatBoost",
]
TUNE_CLASSIFIER = True
CV = 3
CALIBRATE_CLASSIFIER = False

"""
RECOMMENDATION: DO NOT EDIT THE VARIABLES BELOW THIS COMMENT
"""
DECISION_TREE_PARAM_GRID = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 15, 20, 30, 40, 50, 70, 90, 120, 150],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_leaf_nodes': [None, 100, 200, 300, 400, 500]
}

KNN_PARAM_GRID = {
    'n_neighbors': [2, 3, 5],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 50, 80]
}

MLP_PARAM_GRID = {
    'hidden_layer_sizes': [(5, 1), (10, 1), (5, 2), (10, 2), (100,)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [1e-5, 0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive'],
}

NAIVE_BAYES_PARAM_GRID = {
    'var_smoothing': [1e-9, 1e-5]
}

RANDOM_FOREST_PARAM_GRID = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 20, 50, 100],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_leaf_nodes': [None, 100, 250, 500],
    'max_features': ['auto', 'sqrt', 'log2']
}

SVM_PARAM_GRID = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

XGBOOST_PARAM_GRID = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [None, 20, 50, 100],
    'objective': ['binary:logistic', 'binary:hinge']
}

LR_PARAM_GRID = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

LGBM_PARAM_GRID = {
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
}

CATBOOST_PARAM_GRID = {
    'depth': [4, 5, 6, 7, 8, 9, 10],
    'iterations': [100, 250, 500, 1000],
    'learning_rate': [0.001, 0.01, 0.02, 0.03, 0.04],
    'l2_leaf_reg': [1.5, 3, 5, 10, 100],
    'border_count': [5, 10, 20, 32, 50, 100, 200],
    'leaf_estimation_method': [None, 'Newton', 'Gradient']
}
