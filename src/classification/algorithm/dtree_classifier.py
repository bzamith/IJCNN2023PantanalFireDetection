"""
Module which contains the DecisionTreeClassifier class
It contains the required methods to extract train a Decision Tree model
"""
from sklearn import tree

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class DecisionTreeClassifier(Classifier):
    """
    The DecisionTreeClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.DTREE
    base_estimator = tree.DecisionTreeClassifier(random_state=cfg.SEED)
    param_grid = clfcfg.DECISION_TREE_PARAM_GRID
    classifier = None
