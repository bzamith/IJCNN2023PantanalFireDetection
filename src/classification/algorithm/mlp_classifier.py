"""
Module which contains the MLPClassifier class
It contains the required methods to extract train a MLP model
"""
from sklearn import neural_network

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class MLPClassifier(Classifier):
    """
    The MLPClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.MLP
    base_estimator = neural_network.MLPClassifier(max_iter=clfcfg.NB_EPOCHS, random_state=cfg.SEED)
    param_grid = clfcfg.MLP_PARAM_GRID
    classifier = None
