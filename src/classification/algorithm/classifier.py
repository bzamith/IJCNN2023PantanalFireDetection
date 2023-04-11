"""Module which contains the Classifier class"""
import copy
from typing import Any, List

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

import config.classification_settings as clfcfg
import config.general_settings as cfg

from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum


class Classifier:
    """
    The Classifier abstract entity
    This is the abstract class Specific machine learning classifiers should extend this
    """

    algorithm: ClassificationAlgorithmEnum
    base_estimator: Any
    classifier: Any
    param_grid: dict

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Trains the classifier
        :param X: The attribute space for training the algorithm
        :param y: The target space for training the algorithm
        """
        if self.__class__ == Classifier:
            raise Exception("Class Classifier must not be called directly")
        estimator = copy.deepcopy(self.base_estimator)
        if not clfcfg.TUNE_CLASSIFIER:
            estimator.fit(X, y)
        else:
            estimator = self.__get_tuned_classifier(estimator, X, y)
        if clfcfg.CALIBRATE_CLASSIFIER:
            estimator = self.__get_calibrated_classifier(estimator, X, y)
        self.classifier = estimator

    def predict(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Predicts the target given the attribute space
        :param X: The attribute space for testing the algorithm
        :return: The list of predicted probabilities for target space
        """
        if self.__class__ == Classifier:
            raise Exception("Class Classifier must not be called directly")
        if self.classifier is None:
            raise Exception("You must train the classifier before calling predict method")
        prediction = pd.DataFrame(self.classifier.predict(X))
        probabilities = pd.DataFrame(self.classifier.predict_proba(X))
        return [prediction, probabilities]

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Evaluates the classifier using f1 measure
        :param X: The attribute space for evaluating the algorithm
        :param y: The target space for evaluating the algorithm
        :return: The evaluation measure
        """
        if self.__class__ == Classifier:
            raise Exception("Class Classifier must not be called directly")
        if self.classifier is None:
            raise Exception("You must train the classifier before calling evaluate method")
        y_pred = self.predict(X)[0]
        return f1_score(y, y_pred, average='weighted')

    def __get_tuned_classifier(self,
                               base_estimator: Any,
                               X: pd.DataFrame,
                               y: pd.DataFrame) -> Any:
        """
        Performs tuning given a base estimator, X and y. Returns the best estimator after tuning.
        :param base_estimator: The estimator that will be tuned
        :param X: The attribute space
        :param y: The target space
        :return: The tuned estimator
        """
        search = RandomizedSearchCV(base_estimator,
                                    self.param_grid,
                                    cv=clfcfg.CV,
                                    n_jobs=-1,
                                    random_state=cfg.SEED,
                                    scoring='f1_weighted')
        search.fit(X, y)

        best_estimator = search.best_estimator_
        best_estimator.fit(X, y)
        return best_estimator

    def __get_calibrated_classifier(self,
                                    base_estimator: Any,
                                    X: pd.DataFrame,
                                    y: pd.DataFrame) -> Any:
        """
        Performs calibration given a tuned base estimator, X and y. Returns the best estimator after calibration.
        :param base_estimator: The estimator that will be tuned
        :param X: The attribute space
        :param y: The target space
        :return: The calibrated estimator
        """
        calibration = CalibratedClassifierCV(base_estimator=base_estimator,
                                             cv="prefit",
                                             n_jobs=-1,
                                             ensemble=False)
        calibration.fit(X, y)
        calibrated_classifiers = calibration.calibrated_classifiers_

        if len(calibrated_classifiers) > 1:
            raise Exception("There should be a single calibrated classifier")

        return calibration
