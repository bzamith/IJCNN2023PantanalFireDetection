"""
Module which contains the XGBoostClassifier class
It contains the required methods to extract train a XGBoost model
"""
import xgboost as xgb

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class XGBoostClassifier(Classifier):
    """
    The XGBoostClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.XGBOOST
    base_estimator = xgb.XGBClassifier(random_state=cfg.SEED, use_label_encoder=False)
    param_grid = clfcfg.XGBOOST_PARAM_GRID
    classifier = None
