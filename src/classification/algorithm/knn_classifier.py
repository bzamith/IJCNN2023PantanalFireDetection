"""
Module which contains the KNNClassifier class
It contains the required methods to extract train a KNN model
"""
from sklearn.neighbors import KNeighborsClassifier

import config.classification_settings as clfcfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class KNNClassifier(Classifier):
    """
    The KNNClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.KNN
    base_estimator = KNeighborsClassifier()
    param_grid = clfcfg.KNN_PARAM_GRID
    classifier = None
