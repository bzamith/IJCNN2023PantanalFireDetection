"""
Module which contains the LightGBMClassifier class
It contains the required methods to extract train a LightGBM model
"""
import lightgbm as lgbm

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class LightGBMClassifier(Classifier):
    """
    The LightGBMClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.LIGHTGBM
    base_estimator = lgbm.LGBMClassifier(random_state=cfg.SEED)
    param_grid = clfcfg.LGBM_PARAM_GRID
    classifier = None
