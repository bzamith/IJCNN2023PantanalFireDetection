"""Module which represents a factory for ClassificationAlgorithmEnums"""
from src.classification.algorithm.catboost_classifier import CatBoostClassifier
from src.classification.algorithm.classifier import Classifier
from src.classification.algorithm.dtree_classifier import DecisionTreeClassifier
from src.classification.algorithm.knn_classifier import KNNClassifier
from src.classification.algorithm.lightgbm_classifier import LightGBMClassifier
from src.classification.algorithm.logistic_regression_classifier import LogisticRegressionClassifier
from src.classification.algorithm.mlp_classifier import MLPClassifier
from src.classification.algorithm.nb_classifier import NaiveBayesClassifier
from src.classification.algorithm.rforest_classifier import RandomForestClassifier
from src.classification.algorithm.svm_classifier import SVMClassifier
from src.classification.algorithm.xgboost_classifier import XGBoostClassifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum
from src.exception.not_implemented_exception import NotImplementedException


def get(clf_algorithm: ClassificationAlgorithmEnum) -> Classifier:
    """
    Factory method for Classifier
    :param clf_algorithm: The enum for classification algorithm
    :return: The algorithm object for that given enum
    """
    if not clf_algorithm:
        raise ValueError("Parameter algorithm must not be null")
    if not isinstance(clf_algorithm, ClassificationAlgorithmEnum):
        raise TypeError("Parameter algorithm must be of type ClassificationAlgorithmEnum")
    if clf_algorithm == ClassificationAlgorithmEnum.MLP:
        return MLPClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.KNN:
        return KNNClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.SVM:
        return SVMClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.NB:
        return NaiveBayesClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.DTREE:
        return DecisionTreeClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.RFOREST:
        return RandomForestClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.XGBOOST:
        return XGBoostClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.LR:
        return LogisticRegressionClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.LIGHTGBM:
        return LightGBMClassifier()
    if clf_algorithm == ClassificationAlgorithmEnum.CATBOOST:
        return CatBoostClassifier()
    raise NotImplementedException("No Classifier implemented for classification algorithm {}".format(clf_algorithm.value))
