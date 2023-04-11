"""
Module which contains the SVMClassifier class
It contains the required methods to extract train a SVM model
"""
from sklearn.svm import SVC

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class SVMClassifier(Classifier):
    """
    The SVMClassifier entity
    It extends the abstract Classifier class
    """

    algorithm = ClassificationAlgorithmEnum.SVM
    base_estimator = SVC(probability=True, max_iter=clfcfg.NB_EPOCHS, random_state=cfg.SEED)
    param_grid = clfcfg.SVM_PARAM_GRID
    classifier = None
