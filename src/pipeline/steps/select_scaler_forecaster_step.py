"""
Module which contains the SelectScalerForecasterStep, SelectScalerForecasterStepInput and SelectScalerForecasterStepOutput classes
They contain the required methods to scale the datasets (also known as normalization) and train the forecaster, in order to select
the best combination
"""
from typing import List, Tuple

import pandas as pd

import config.dataset_settings as dscfg
import config.general_settings as cfg

import src.utils.pandas_utils as pdutils
from src.enum.forecasting_algorithms_enum import ForecastingAlgorithmEnum
from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.forecasting.algorithm import forecaster_factory
from src.forecasting.algorithm.forecaster import Forecaster
from src.pipeline.step import Step, StepInput, StepOutput
from src.scaling import scaler_factory
from src.scaling.scaler import Scaler


class SelectScalerForecasterStep(Step):
    """The SelectScalerForecasterStep entity"""

    step_name = "Select Scaler and Forecaster"
    step_description = "Trains combinations of scalers and forecasters and selects the best one"

    def __init__(self, scaling_methods: List[ScalingMethodEnum],
                 forecasting_algorithms: List[ForecastingAlgorithmEnum],
                 X_train: pd.DataFrame,
                 X_validation: pd.DataFrame):
        """
        Class constructor
        :param scaling_methods: the scaling methods that should be trained and validated
        :param forecasting_algorithms: the forecasting algorithms that should be trained and validated
        :param X_train: attributes for training
        :param X_validation: attributes for validation
        """
        self.step_input = SelectScalerForecasterStepInput(scaling_methods,
                                                          forecasting_algorithms,
                                                          X_train,
                                                          X_validation)
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        scaler, forecaster, scaled_X_train, scaled_X_validation, execution_summary = self.__select()

        self.__save_selected_scaler_info(scaler)
        self.__save_selected_forecaster_info(forecaster)

        self.step_output = SelectScalerForecasterStepOutput(scaler,
                                                            forecaster,
                                                            execution_summary,
                                                            scaled_X_train,
                                                            scaled_X_validation)

    def __select(self) -> Tuple[Scaler, Forecaster, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Selects best combination"""
        scaling_methods = self.step_input.scaling_methods
        forecasting_algorithms = self.step_input.forecasting_algorithms

        X_train = self.step_input.X_train
        X_validation = self.step_input.X_validation

        best_forecaster = None
        best_scaler = None
        best_scaler_forecaster_error_metric = None
        best_scaled_X_train = None
        best_scaled_X_validation = None

        scaling_methods_run = []
        forecasting_algorithms_run = []
        error_metrics_run = []

        for scaling_method in scaling_methods:
            scaler = scaler_factory.get(scaling_method)
            scaled_X_train = scaler.fit_scale(X_train)
            scaled_X_validation = scaler.scale(X_validation)
            for forecast_algorithm in forecasting_algorithms:
                forecaster = forecaster_factory.get(forecast_algorithm)
                forecaster.learn(pdutils.delete_columns(scaled_X_train, dscfg.COLUMNS_IGNORE_FOR_ML))
                error_metric = forecaster.evaluate(pdutils.delete_columns(scaled_X_validation, dscfg.COLUMNS_IGNORE_FOR_ML))
                scaling_methods_run.append(scaling_method)
                forecasting_algorithms_run.append(forecast_algorithm)
                error_metrics_run.append(error_metric)
                if best_scaler_forecaster_error_metric is None or error_metric[0] < best_scaler_forecaster_error_metric[0]:
                    best_scaler_forecaster_error_metric = error_metric
                    best_scaler = scaler
                    best_forecaster = forecaster
                    best_scaled_X_train = scaled_X_train
                    best_scaled_X_validation = scaled_X_validation

        execution_summary = pd.DataFrame({
            'scaling_method': scaling_methods_run,
            'forecast_algorithm': forecasting_algorithms_run,
            'error_metric': error_metrics_run
        })
        execution_summary.to_csv(cfg.ASSETS_DIR + "select_scaler_forecaster_execution_summary.csv")

        return best_scaler, best_forecaster, best_scaled_X_train, best_scaled_X_validation, execution_summary

    def __save_selected_scaler_info(self, scaler: Scaler) -> None:
        with open(cfg.ASSETS_DIR + "scaler.txt", 'w') as f:
            f.write("Selected best scaler: ")
            f.write("\n")
            f.write(scaler.method.value)

    def __save_selected_forecaster_info(self, forecaster: Forecaster) -> None:
        with open(cfg.ASSETS_DIR + "forecaster.txt", 'w') as f:
            f.write("Selected best forecaster: ")
            f.write("\n")
            f.write(forecaster.algorithm.value)


class SelectScalerForecasterStepInput(StepInput):
    """Input for SelectScalerForecasterStep"""

    scaling_methods: List[ScalingMethodEnum]
    forecasting_algorithms: List[ForecastingAlgorithmEnum]
    X_train: pd.DataFrame
    X_validation: pd.DataFrame

    def __init__(self,
                 scaling_methods: List[ScalingMethodEnum],
                 forecasting_algorithms: List[ForecastingAlgorithmEnum],
                 X_train: pd.DataFrame,
                 X_validation: pd.DataFrame
                 ):
        """
        Class constructor
        :param scaling_methods: the scaling methods that should be trained and validated
        :param forecasting_algorithms: the forecasting algorithms that should be trained and validated
        :param X_train: attributes for training
        :param X_validation: attributes for validation
        :param X_train_validation: attributes for training plus validation
        """
        self.scaling_methods = scaling_methods
        self.forecasting_algorithms = forecasting_algorithms
        self.X_train = X_train
        self.X_validation = X_validation


class SelectScalerForecasterStepOutput(StepOutput):
    """Output for SelectScalerForecasterStep"""

    scaler: Scaler
    forecaster: Forecaster
    execution_summary: pd.DataFrame
    scaled_X_train: pd.DataFrame
    scaled_X_validation: pd.DataFrame

    def __init__(self,
                 scaler: Scaler,
                 forecaster: Forecaster,
                 execution_summary: pd.DataFrame,
                 scaled_X_train: pd.DataFrame,
                 scaled_X_validation: pd.DataFrame,
                 ):
        """
        Class constructor
        :param scaler: the trained scaler
        :param scaler: the trained forecaster
        :param execution_summary: the execution summary
        :param scaled_X_train: scaled dataset with attributes for training
        :param scaled_X_validation: scaled dataset with attributes for validation
        """
        self.scaler = scaler
        self.forecaster = forecaster
        self.execution_summary = execution_summary
        self.scaled_X_train = scaled_X_train
        self.scaled_X_validation = scaled_X_validation
