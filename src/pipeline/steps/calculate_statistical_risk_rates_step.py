"""
Module which contains the CalculateStatisticalRiskRatesStep, CalculateStatisticalRiskRatesStepInput and CalculateStatisticalRiskRatesStepOutput classes
They contain the required methods to calculate the statistical/known risk rates such as FMA, Angstron and so on
"""
from typing import List

import pandas as pd

import config.dataset_settings as dscfg
import config.general_settings as cfg

import src.utils.pandas_utils as pdutils
from src.enum.risk_rate_algorithms_enum import RiskRateAlgorithmEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.pipeline.step import Step, StepInput, StepOutput
from src.risk_rate.algorithm.risk_rate_algorithm_factory import get
from src.utils.dataset_columns_utils import get_column_names_suffix


class CalculateStatisticalRiskRatesStep(Step):
    """The CalculateStatisticalRiskRatesStep entity"""

    step_name = "Calculate Statistical Risk Rates"
    step_description = "Calculate the risk rates for the predicted results, given statistical methods"

    def __init__(self, input_dataset: pd.DataFrame,
                 X_test: pd.DataFrame,
                 forecasted_X_tests: List[pd.DataFrame]):
        """
        Class constructor
        :param input_dataset: the initial and prepared dataset
        :param X_test: attributes for testing
        :param forecasted_X_tests: the list of forecasted datasets (one per day in the future)
        """
        self.step_input = CalculateStatisticalRiskRatesStepInput(input_dataset,
                                                                 X_test,
                                                                 forecasted_X_tests)
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        input_dataset = self.step_input.input_dataset
        X_test = self.step_input.X_test
        forecasted_X_tests = self.step_input.forecasted_X_tests
        dates_x_test = pdutils.select_columns(X_test, dscfg.DATE_COLUMN_NAME)

        # Select values from input_dataset that are in x_test (date column)
        column_names = list(input_dataset.columns.values)
        joined = pdutils.join_inner_by_date([input_dataset, dates_x_test])
        input_dataset = pdutils.select_columns(joined, column_names)

        output_dataset = pdutils.select_columns(input_dataset, dscfg.DATE_COLUMN_NAME)

        present_column_name = get_column_names_suffix(dscfg.PRESENT_COLUMN_NAME, "")[0]
        forecasted_column_names = get_column_names_suffix(dscfg.FORECASTED_COLUMN_NAME, "")

        for risk_rate_algorithm in RiskRateAlgorithmEnum:
            try:
                algorithm = get(risk_rate_algorithm)
                dataset = algorithm.calculate_for_dataset(input_dataset)
                dataset = pdutils.add_prefix_to_column_names(dataset, present_column_name)
                output_dataset = pdutils.join_dataframes_y_wise([output_dataset, dataset])
                for i, forecasted_dataset in enumerate(forecasted_X_tests):
                    forecasted_column_name = forecasted_column_names[i]
                    dataset = algorithm.calculate_for_dataset(forecasted_dataset)
                    dataset = pdutils.add_prefix_to_column_names(dataset, forecasted_column_name)
                    output_dataset = pdutils.join_dataframes_y_wise([output_dataset, dataset])
            except NotImplementedException:
                pass

        output_dataset.to_csv(cfg.DATA_GENERATED_DIR + "statistical_risk_rates.csv")
        self.step_output = CalculateStatisticalRiskRatesStepOutput(output_dataset)


class CalculateStatisticalRiskRatesStepInput(StepInput):
    """Input for CalculateStatisticalRiskRatesStep"""

    input_dataset: pd.DataFrame
    X_test: pd.DataFrame
    forecasted_X_tests: List[pd.DataFrame]

    def __init__(self,
                 input_dataset: pd.DataFrame,
                 X_test: pd.DataFrame,
                 forecasted_X_tests: List[pd.DataFrame]):
        """
        Class constructor
        :param input_dataset: the initial and prepared dataset
        :param X_test: attributes for testing
        :param forecasted_X_tests: the list of forecasted datasets (one per day in the future)
        """
        self.input_dataset = input_dataset
        self.X_test = X_test
        self.forecasted_X_tests = forecasted_X_tests


class CalculateStatisticalRiskRatesStepOutput(StepOutput):
    """Output for CalculateStatisticalRiskRatesStep"""

    statistical_risk_rates_dataset: pd.DataFrame

    def __init__(self, statistical_risk_rates_dataset: pd.DataFrame):
        """
        Class constructor
        :param statistical_risk_rates_dataset: the dataset with statistical risk rates for dataset
        """
        self.statistical_risk_rates_dataset = statistical_risk_rates_dataset
