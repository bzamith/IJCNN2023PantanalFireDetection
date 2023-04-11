"""
Module which contains the CatBoostClassifier class
It contains the required methods to extract train a CatBoost model
"""
import catboost as ctb

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class CatBoostClassifier(Classifier):
    """
    The CatBoostClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.CATBOOST
    base_estimator = ctb.CatBoostClassifier(random_state=cfg.SEED)
    param_grid = clfcfg.CATBOOST_PARAM_GRID
    classifier = None
