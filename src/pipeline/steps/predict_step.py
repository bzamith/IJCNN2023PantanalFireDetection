"""
Module which contains the PredictStep, PredictStepInput and PredictStepOutput classes
They contain the required methods to run the prediction algorithm (ml)
"""
from typing import List

import pandas as pd

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils
from src.classification.algorithm.classifier import Classifier
from src.pipeline.step import Step, StepInput, StepOutput
from src.utils.dataset_columns_utils import get_column_name_prefix, get_column_names_prefix


class PredictStep(Step):
    """The PredictStep entity"""

    step_name = "Predict"
    step_description = "Predict probabilities for present and forecasted data"

    def __init__(self, classifier: Classifier,
                 scaled_X_validation: pd.DataFrame,
                 forecasted_scaled_X_validations: List[pd.DataFrame]):
        """
        Class constructor
        :param classifier: the trained classifier
        :param scaled_X_validation: the scaled feature space for validation
        :param forecasted_scaled_X_validations: the list of scaled forecasted datasets (one per day in the future)
        """
        self.step_input = PredictStepInput(classifier, scaled_X_validation, forecasted_scaled_X_validations)
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        classifier = self.step_input.classifier
        scaled_X_validation = self.step_input.scaled_X_validation
        forecasted_scaled_X_validations = self.step_input.forecasted_scaled_X_validations

        scaled_X_validation_updated = pdutils.delete_columns(scaled_X_validation, dscfg.COLUMNS_IGNORE_FOR_ML)
        probs_present = classifier.predict(scaled_X_validation_updated)[1].iloc[:, 1]

        probs_forecasted_list = []
        for forecasted_scaled_X_validation in forecasted_scaled_X_validations:
            forecasted_scaled_X_validation_updated = pdutils.delete_columns(forecasted_scaled_X_validation, dscfg.COLUMNS_IGNORE_FOR_ML)
            probs = classifier.predict(forecasted_scaled_X_validation_updated)[1].iloc[:, 1]
            probs_forecasted_list.append(probs)

        predicted_dataset = self.__create_probs_dataset(probs_present, probs_forecasted_list, scaled_X_validation)

        self.step_output = PredictStepOutput(predicted_dataset)

    def __create_probs_dataset(self,
                               probs_present: List[float],
                               probs_forecasted_list: List[List[float]],
                               X_validation: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the dataset with probabilities
        :param probs_present: the list of probabilities for present
        :param probs_forecasted_list: the list of probabilities for each day in the future
        :param X_validation: the X_validation dataset, for checking the dates
        :return: the dataframe with probabilities
        """
        probs_name_present = get_column_name_prefix(dscfg.PRESENT_COLUMN_NAME, dscfg.PROBS_PREFIX)
        probs_name_forecasted = get_column_names_prefix(dscfg.FORECASTED_COLUMN_NAME, dscfg.PROBS_PREFIX)
        dates = pdutils.select_columns(X_validation, dscfg.DATE_COLUMN_NAME, reset_row_indexes=True)

        data_map = {
            probs_name_present: probs_present
        }

        for i in range(0, len(probs_forecasted_list)):
            key = probs_name_forecasted[i]
            data_map[key] = probs_forecasted_list[i]

        return pdutils.join_dataframes_y_wise([dates, pd.DataFrame(data_map)])


class PredictStepInput(StepInput):
    """Input for PredictStep"""

    classifier: Classifier
    scaled_X_validation: pd.DataFrame
    forecasted_scaled_X_validations: List[pd.DataFrame]

    def __init__(self,
                 classifier: Classifier,
                 scaled_X_validation: pd.DataFrame,
                 forecasted_scaled_X_validations: List[pd.DataFrame]):
        """
        Class constructor
        :param classifier: the trained classifier
        :param scaled_X_validation: the scaled feature space for validation
        :param forecasted_scaled_X_validations: the list of scaled forecasted datasets (one per day in the future)
        """
        self.classifier = classifier
        self.scaled_X_validation = scaled_X_validation
        self.forecasted_scaled_X_validations = forecasted_scaled_X_validations


class PredictStepOutput(StepOutput):
    """Output for PredictStep"""

    predicted_dataset: pd.DataFrame

    def __init__(self, predicted_dataset: pd.DataFrame):
        """
        Class constructor
        :param predicted_dataset: the dataset with the predictions
        """
        self.predicted_dataset = predicted_dataset
