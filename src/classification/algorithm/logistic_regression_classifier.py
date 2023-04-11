"""
Module which contains the LogisticRegressionClassifier class
It contains the required methods to extract train a Logistic Regression model
"""
from sklearn.linear_model import LogisticRegression

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class LogisticRegressionClassifier(Classifier):
    """
    The LogisticRegressionClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.LR
    base_estimator = LogisticRegression(random_state=cfg.SEED)
    param_grid = clfcfg.LR_PARAM_GRID
    classifier = None
