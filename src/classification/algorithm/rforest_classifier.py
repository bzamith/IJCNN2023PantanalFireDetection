"""
Module which contains the RandomForest class
It contains the required methods to extract train a Random Forest model
"""
from sklearn import ensemble

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class RandomForestClassifier(Classifier):
    """
    The RandomForestClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.RFOREST
    base_estimator = ensemble.RandomForestClassifier(random_state=cfg.SEED)
    param_grid = clfcfg.RANDOM_FOREST_PARAM_GRID
    classifier = None
