from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn import tree

import config.classification_settings as clfcfg

from src.classification.algorithm.dtree_classifier import DecisionTreeClassifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


def test_attributes():
    classifier = DecisionTreeClassifier()
    assert classifier.algorithm == ClassificationAlgorithmEnum.DTREE
    assert isinstance(classifier.base_estimator, tree.DecisionTreeClassifier)
    assert classifier.param_grid == clfcfg.DECISION_TREE_PARAM_GRID
    assert classifier.classifier is None


@mock.patch('src.classification.algorithm.classifier.RandomizedSearchCV')
@mock.patch('src.classification.algorithm.classifier.CalibratedClassifierCV')
def test_train_cv(mock_calibration, mock_search_cv):
    actual_tune_classifier = clfcfg.TUNE_CLASSIFIER
    actual_calibrate_classifier = clfcfg.CALIBRATE_CLASSIFIER

    clfcfg.TUNE_CLASSIFIER = True
    clfcfg.CALIBRATE_CLASSIFIER = True

    valid_x = pd.DataFrame({'col': ['0', '1']})
    valid_y = pd.DataFrame({'col': ['2', '3']})

    classifier = DecisionTreeClassifier()

    mock_classifier_best_estimator = MagicMock(spec=tree.DecisionTreeClassifier)
    mock_search_cv.best_estimator_.return_value = mock_classifier_best_estimator

    mock_classifier_calibrated_classifier = MagicMock(spec=tree.DecisionTreeClassifier)
    mock_calibration.calibrated_classifiers_.return_value = [mock_classifier_calibrated_classifier]

    classifier.train(valid_x, valid_y)

    assert mock_search_cv.call_args[1]['cv'] == 3
    assert mock_search_cv.call_args[1]['n_jobs'] == -1
    assert mock_search_cv.call_args[1]['scoring'] == 'f1_weighted'

    assert mock_calibration.call_args[1]['cv'] == "prefit"
    assert mock_calibration.call_args[1]['n_jobs'] == -1
    assert mock_calibration.call_args[1]['ensemble'] == False

    clfcfg.TUNE_CLASSIFIER = actual_tune_classifier
    clfcfg.CALIBRATE_CLASSIFIER = actual_calibrate_classifier


@mock.patch('sklearn.tree.DecisionTreeClassifier.fit')
@mock.patch('src.classification.algorithm.classifier.CalibratedClassifierCV')
def test_train_no_cv(mock_calibration, mock_classifier_fit):
    actual_tune_classifier = clfcfg.TUNE_CLASSIFIER
    actual_calibrate_classifier = clfcfg.CALIBRATE_CLASSIFIER

    clfcfg.TUNE_CLASSIFIER = False
    clfcfg.CALIBRATE_CLASSIFIER = True

    valid_x = pd.DataFrame({'col': ['value']})
    valid_y = pd.DataFrame({'col': ['0']})

    mock_classifier_calibrated_classifier = MagicMock(spec=tree.DecisionTreeClassifier)
    mock_calibration.calibrated_classifiers_.return_value = [mock_classifier_calibrated_classifier]

    mock_classifier_fit.return_value.fit.return_value = "return"

    DecisionTreeClassifier().train(valid_x, valid_y)

    mock_classifier_fit.assert_called_with(valid_x, valid_y)

    clfcfg.TUNE_CLASSIFIER = actual_tune_classifier
    clfcfg.CALIBRATE_CLASSIFIER = actual_calibrate_classifier


@mock.patch('sklearn.tree.DecisionTreeClassifier.predict_proba')
@mock.patch('sklearn.tree.DecisionTreeClassifier.predict')
def test_predict(mock_classifier_predict, mock_classifier_predict_proba):
    valid_x = pd.DataFrame({'col': ['value']})
    expected_output_predict = pd.DataFrame({'col': ['0']})
    expected_output_prob = pd.DataFrame({'col': ['0.1']})

    mock_classifier_predict.return_value = expected_output_predict
    mock_classifier_predict_proba.return_value = expected_output_prob

    classifier = DecisionTreeClassifier()
    classifier.classifier = tree.DecisionTreeClassifier()

    output = classifier.predict(valid_x)

    mock_classifier_predict.assert_called_with(valid_x)
    mock_classifier_predict_proba.assert_called_with(valid_x)

    assert len(output) == 2
    assert output[0].equals(expected_output_predict)
    assert output[1].equals(expected_output_prob)


def test_predict_none_classifier():
    valid_x = pd.DataFrame({'col': ['value']})
    with pytest.raises(Exception) as e_info:
        DecisionTreeClassifier().predict(valid_x)
    assert str(e_info.value) == "You must train the classifier before calling predict method"


@mock.patch('src.classification.algorithm.classifier.f1_score')
@mock.patch('src.classification.algorithm.classifier.Classifier.predict')
def test_evaluate(mock_classifier_predict, mock_f1_score):
    valid_x = pd.DataFrame({'col': ['value']})
    valid_y = pd.DataFrame({'col': ['1']})
    expected_output_predict = pd.DataFrame({'col': ['0']})
    expected_output_f1_score = 0.87

    mock_classifier_predict.return_value = [expected_output_predict, pd.DataFrame({'col': ['1']})]
    mock_f1_score.return_value = expected_output_f1_score

    classifier = DecisionTreeClassifier()
    classifier.classifier = tree.DecisionTreeClassifier()

    output = classifier.evaluate(valid_x, valid_y)

    mock_classifier_predict.assert_called_with(valid_x)
    mock_f1_score.assert_called_with(valid_y, expected_output_predict, average='weighted')

    assert output == expected_output_f1_score


def test_evaluate_none_classifier():
    valid_x = pd.DataFrame({'col': ['value']})
    valid_y = pd.DataFrame({'col': ['1']})
    with pytest.raises(Exception) as e_info:
        DecisionTreeClassifier().evaluate(valid_x, valid_y)
    assert str(e_info.value) == "You must train the classifier before calling evaluate method"
