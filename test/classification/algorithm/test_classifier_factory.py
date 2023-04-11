from unittest.mock import MagicMock

import pytest

from src.classification.algorithm import classifier_factory
from src.classification.algorithm.catboost_classifier import CatBoostClassifier
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


def test_get_mlp():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.MLP)
    assert isinstance(classifier, MLPClassifier)


def test_get_invalid():
    with pytest.raises(TypeError) as e_info:
        classifier_factory.get("xxx")
    assert str(
        e_info.value) == "Parameter algorithm must be of type ClassificationAlgorithmEnum"


def test_get_none():
    with pytest.raises(ValueError) as e_info:
        classifier_factory.get(None)
    assert str(e_info.value) == "Parameter algorithm must not be null"


def test_get_knn():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.KNN)
    assert isinstance(classifier, KNNClassifier)


def test_get_dtree():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.DTREE)
    assert isinstance(classifier, DecisionTreeClassifier)


def test_get_rforest():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.RFOREST)
    assert isinstance(classifier, RandomForestClassifier)


def test_get_svm():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.SVM)
    assert isinstance(classifier, SVMClassifier)


def test_get_nb():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.NB)
    assert isinstance(classifier, NaiveBayesClassifier)


def test_get_xgboost():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.XGBOOST)
    assert isinstance(classifier, XGBoostClassifier)


def test_get_lr():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.LR)
    assert isinstance(classifier, LogisticRegressionClassifier)


def test_get_lightgbm():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.LIGHTGBM)
    assert isinstance(classifier, LightGBMClassifier)


def test_get_catboost():
    classifier = classifier_factory.get(ClassificationAlgorithmEnum.CATBOOST)
    assert isinstance(classifier, CatBoostClassifier)


def test_get_not_implemented():
    mock = MagicMock(spec=ClassificationAlgorithmEnum, name="Dummy", value="DummyValue")
    with pytest.raises(NotImplementedException) as e_info:
        classifier_factory.get(mock)
    assert str(e_info.value) == "No Classifier implemented for classification algorithm DummyValue"
