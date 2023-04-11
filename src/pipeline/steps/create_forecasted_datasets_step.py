"""
Module which contains the CreateForecastedDatasetsStep, CreateForecastedDatasetsStepInput and CreateForecastedDatasetsStepOutput classes
They contain the required methods to forecast the next X days climatic data
"""
from typing import List

import pandas as pd

from src.forecasting.algorithm.forecaster import Forecaster
from src.forecasting.forecast_operations import create_forecasted_datasets
from src.pipeline.step import Step, StepInput, StepOutput
from src.scaling.scaler import Scaler


class CreateForecastedDatasetsStep(Step):
    """The CreateForecastedDatasetsStep entity"""

    step_name = "Create Forecasted Datasets"
    step_description = "Create and save the forecasted datasets"

    def __init__(self, scaler: Scaler,
                 forecaster: Forecaster,
                 scaled_X_train: pd.DataFrame,
                 X_test: pd.DataFrame):
        """
        Class constructor
        :param scaler: the trained scaler
        :param forecaster: the trained forecaster
        :param scaled_X_train: scaled dataset with attributes for train
        :param X_test: attributes for test
        """
        self.step_input = CreateForecastedDatasetsStepInput(
            scaler, forecaster, scaled_X_train, X_test
        )
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        scaler = self.step_input.scaler
        forecaster = self.step_input.forecaster
        scaled_X_train = self.step_input.scaled_X_train
        X_test = self.step_input.X_test

        scaled_X_test = scaler.scale(X_test)

        forecasted_scaled_X_tests = create_forecasted_datasets(forecaster, scaled_X_train, scaled_X_test)

        forecasted_X_tests = []
        for forecasted_scaled_X_test in forecasted_scaled_X_tests:
            forecasted_X_tests.append(scaler.descale(forecasted_scaled_X_test))

        self.step_output = CreateForecastedDatasetsStepOutput(
            scaled_X_test,
            forecasted_X_tests,
            forecasted_scaled_X_tests
        )


class CreateForecastedDatasetsStepInput(StepInput):
    """Input for CreateForecastedDatasetsStep"""

    scaler: Scaler
    forecaster: Forecaster
    scaled_X_train: pd.DataFrame
    X_test: pd.DataFrame

    def __init__(self, scaler: Scaler,
                 forecaster: Forecaster,
                 scaled_X_train: pd.DataFrame,
                 X_test: pd.DataFrame):
        """
        Class constructor
        :param scaler: the trained scaler
        :param forecaster: the trained forecaster
        :param scaled_X_train: scaled dataset with attributes for train
        :param X_test: attributes for test
        """
        self.scaler = scaler
        self.forecaster = forecaster
        self.scaled_X_train = scaled_X_train
        self.X_test = X_test


class CreateForecastedDatasetsStepOutput(StepOutput):
    """Output for CreateForecastedDatasetsStep"""

    scaled_X_test: pd.DataFrame
    forecasted_X_tests: List[pd.DataFrame]
    forecasted_scaled_X_tests: List[pd.DataFrame]

    def __init__(self,
                 scaled_X_test: pd.DataFrame,
                 forecasted_X_tests: List[pd.DataFrame],
                 forecasted_scaled_X_tests: List[pd.DataFrame]):
        """
        Class constructor
        :param scaled_X_test: the scaled feature space for test
        :param forecasted_X_tests: the list of forecasted datasets (one per day in the future)
        :param forecasted_scaled_X_tests: the list of scaled forecasted datasets (one per day in the future)
        """
        self.scaled_X_test = scaled_X_test
        self.forecasted_X_tests = forecasted_X_tests
        self.forecasted_scaled_X_tests = forecasted_scaled_X_tests
