"""
Module which contains the NaiveBayesClassifier class
It contains the required methods to extract train a NB model
"""
from sklearn.naive_bayes import GaussianNB

import config.classification_settings as clfcfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class NaiveBayesClassifier(Classifier):
    """The NaiveBayesClassifier entity"""

    algorithm = ClassificationAlgorithmEnum.NB
    base_estimator = GaussianNB()
    param_grid = clfcfg.NAIVE_BAYES_PARAM_GRID
    classifier = None
