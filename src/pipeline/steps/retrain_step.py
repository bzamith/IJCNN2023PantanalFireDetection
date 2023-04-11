"""
Module which contains the RetrainStep, RetrainStepInput and RetrainStepOutput classes
They contain the required methods to retrain the models
"""
from typing import Tuple

import pandas as pd

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils
from src.classification.algorithm.classifier import Classifier
from src.forecasting.algorithm.forecaster import Forecaster
from src.pipeline.step import Step, StepInput, StepOutput
from src.sampling.sampler import Sampler
from src.scaling.scaler import Scaler


class RetrainStep(Step):
    """The RetrainStep entity"""

    step_name = "Retrain"
    step_description = "Retrain models"

    def __init__(self, scaler: Scaler,
                 forecaster: Forecaster,
                 sampler: Sampler,
                 classifier: Classifier,
                 X_train_validation: pd.DataFrame,
                 y_train_validation: pd.DataFrame):
        """
        Class constructor
        :param scaler: the trained scaler
        :param forecaster: the trained forecaster
        :param sampler: the trained sampler
        :param classifier: the trained classifier
        :param X_train_validation: attributes for training plus validation
        :param y_train_validation: targets for training plus validation
        """
        self.step_input = RetrainStepInput(
            scaler, forecaster, sampler, classifier,
            X_train_validation, y_train_validation
        )
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        i_scaler = self.step_input.scaler
        i_forecaster = self.step_input.forecaster
        i_sampler = self.step_input.sampler
        i_classifier = self.step_input.classifier
        X_train_validation = self.step_input.X_train_validation
        y_train_validation = self.step_input.y_train_validation

        scaler, scaled_X_train_validation = self.__re_train_scaler(i_scaler, X_train_validation)
        forecaster = self.__re_train_forecaster(i_forecaster, scaled_X_train_validation)
        sampler, sampled_scaled_X_train_validation, sampled_y_train_validation = self.__re_train_sampler(
            i_sampler, scaled_X_train_validation, y_train_validation
        )
        classifier = self.__re_train_classifier(i_classifier, sampled_scaled_X_train_validation, sampled_y_train_validation)

        self.step_output = RetrainStepOutput(
            scaler, forecaster, sampler, classifier,
            scaled_X_train_validation,
            sampled_scaled_X_train_validation, sampled_y_train_validation
        )

    def __re_train_scaler(self, scaler: Scaler, X_train_validation: pd.DataFrame) -> Tuple[Scaler, pd.DataFrame]:
        """Re-trains the best combination"""
        scaled_X_train_validation = scaler.fit_scale(X_train_validation)

        return scaler, scaled_X_train_validation

    def __re_train_forecaster(self, forecaster: Forecaster, scaled_X_train_validation: pd.DataFrame) -> Forecaster:
        forecaster.learn(pdutils.delete_columns(scaled_X_train_validation, dscfg.COLUMNS_IGNORE_FOR_ML))

        return forecaster

    def __re_train_sampler(self, sampler: Sampler,
                           scaled_X_train_validation: pd.DataFrame,
                           y_train_validation: pd.DataFrame) -> Tuple[Sampler, pd.DataFrame, pd.DataFrame]:
        """Re-trains the best combination"""
        scaled_X_train_validation = pdutils.select_columns(scaled_X_train_validation, dscfg.COLUMNS_SAMPLING)
        sampled_scaled_X_train_validation, sampled_y_train_validation = sampler.fit_sample(scaled_X_train_validation, y_train_validation)

        return sampler, sampled_scaled_X_train_validation, sampled_y_train_validation

    def __re_train_classifier(self, classifier: Classifier,
                              sampled_scaled_X_train_validation: pd.DataFrame,
                              sampled_y_train_validation: pd.DataFrame) -> Classifier:
        """Re-trains the best combination"""
        classifier.train(sampled_scaled_X_train_validation, sampled_y_train_validation)

        return classifier


class RetrainStepInput(StepInput):
    """Input for RetrainStep"""

    scaler: Scaler
    forecaster: Forecaster
    sampler: Sampler
    classifier: Classifier
    X_train_validation: pd.DataFrame
    y_train_validation: pd.DataFrame

    def __init__(self, scaler: Scaler,
                 forecaster: Forecaster,
                 sampler: Sampler,
                 classifier: Classifier,
                 X_train_validation: pd.DataFrame,
                 y_train_validation: pd.DataFrame):
        """
        Class constructor
        :param scaler: the trained scaler
        :param forecaster: the trained forecaster
        :param sampler: the trained sampler
        :param classifier: the trained classifier
        :param X_train_validation: attributes for training plus validation
        :param y_train_validation: targets for training plus validation
        """
        self.scaler = scaler
        self.forecaster = forecaster
        self.sampler = sampler
        self.classifier = classifier
        self.X_train_validation = X_train_validation
        self.y_train_validation = y_train_validation


class RetrainStepOutput(StepOutput):
    """Output for RetrainStep"""

    scaler: Scaler
    forecaster: Forecaster
    sampler: Sampler
    classifier: Classifier
    scaled_X_train_validation: pd.DataFrame
    sampled_scaled_X_train_validation: pd.DataFrame
    sampled_y_train_validation: pd.DataFrame

    def __init__(self, scaler: Scaler,
                 forecaster: Forecaster,
                 sampler: Sampler,
                 classifier: Classifier,
                 scaled_X_train_validation: pd.DataFrame,
                 sampled_scaled_X_train_validation: pd.DataFrame,
                 sampled_y_train_validation: pd.DataFrame
                 ):
        """
        Class constructor
        :param scaler: the trained scaler
        :param forecaster: the trained forecaster
        :param sampler: the trained sampler
        :param classifier: the trained classifier
        :param scaled_X_train_validation: scaled attributes for training plus validation
        :param sampled_scaled_X_train_validation: sampled and scaled attributes for training plus validation
        :param sampled_y_train_validation: sampled targets for training plus validation
        """
        self.scaler = scaler
        self.forecaster = forecaster
        self.sampler = sampler
        self.classifier = classifier
        self.scaled_X_train_validation = scaled_X_train_validation
        self.sampled_scaled_X_train_validation = sampled_scaled_X_train_validation
        self.sampled_y_train_validation = sampled_y_train_validation
