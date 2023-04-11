"""
Module which contains the CreateResultsOutputDatasetStep, CreateResultsOutputDatasetStepInput and CreateResultsOutputDatasetStepOutput classes
They contain the required methods to create the final output dataset
"""
import pandas as pd

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils
from src.enum.risk_rate_index_enum import RiskRateIndexEnum
from src.pipeline.step import Step, StepInput, StepOutput
from src.utils.dataset_columns_utils import get_column_names_suffix


class CreateResultsOutputDatasetStep(Step):
    """The CreateResultsOutputDatasetStep entity"""

    step_name = "Create Results Output Dataset"
    step_description = "Create and save output dataset"

    def __init__(self, input_dataset: pd.DataFrame,
                 predicted_dataset: pd.DataFrame,
                 predicted_risk_rates_dataset: pd.DataFrame,
                 statistical_risk_rates_dataset: pd.DataFrame):
        """
        Class constructor
        :param input_dataset: the initial and prepared dataset
        :param predicted_dataset: the dataset with predicted probabilities
        :param predicted_risk_rates_dataset: the dataset with risk rates for predicted dataset, given the thresholds
        :param statistical_risk_rates_dataset: the dataset with statistical risk rates for dataset
        """
        self.step_input = CreateResultsOutputDatasetStepInput(
            input_dataset, predicted_dataset, predicted_risk_rates_dataset, statistical_risk_rates_dataset
        )
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        input_dataset = self.step_input.input_dataset
        predicted_dataset = self.step_input.predicted_dataset
        predicted_risk_rates_dataset = self.step_input.predicted_risk_rates_dataset
        try:
            predicted_risk_rates_dataset = pdutils.delete_columns(predicted_risk_rates_dataset, dscfg.HOTSPOT_IDENTIFIED_COLUMN_NAME)
        except KeyError:
            pass
        statistical_risk_rates_dataset = self.step_input.statistical_risk_rates_dataset

        dataframes = [input_dataset, predicted_dataset, predicted_risk_rates_dataset, statistical_risk_rates_dataset]

        output_dataset = dataframes[0]
        for i in range(1, len(dataframes)):
            output_dataset = pdutils.join_inner_by_date([output_dataset, dataframes[i]])

        output_dataset = pdutils.remove_duplicated_columns(output_dataset)
        factorized_risk_rate = self.__factorize_risk_rate(output_dataset)
        output_dataset = pdutils.join_dataframes_y_wise([output_dataset, factorized_risk_rate])
        self.step_output = CreateResultsOutputDatasetStepOutput(output_dataset)

    def __factorize_risk_rate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Gets the risk rate factors (PEQUENO, MEDIO, etc)
        :param dataset: the input dataset
        :return: the dataset with risk rates
        """
        dict_values = {}
        for risk_rate in RiskRateIndexEnum:
            dict_values[risk_rate.value] = risk_rate.factor_value

        present_index_column_names = get_column_names_suffix(dscfg.PRESENT_COLUMN_NAME, dscfg.RISK_RATE_INDEX_COLUMN_NAME_SUFFIX)
        forecasted_index_column_names = get_column_names_suffix(dscfg.FORECASTED_COLUMN_NAME, dscfg.RISK_RATE_INDEX_COLUMN_NAME_SUFFIX)

        dataset = pdutils.select_columns(
            dataset,
            present_index_column_names + forecasted_index_column_names
        )
        dataset_present = pdutils.select_columns(dataset, present_index_column_names).applymap(lambda x: dict_values[x])
        dataset_forecasted = pdutils.select_columns(dataset, forecasted_index_column_names).applymap(lambda x: dict_values[x])
        dataset = pdutils.join_dataframes_y_wise([dataset_present, dataset_forecasted])
        dataset.columns = get_column_names_suffix(dscfg.PRESENT_COLUMN_NAME, dscfg.RISK_RATE_FACTOR_VALUE_COLUMN_NAME_SUFFIX) + \
                          get_column_names_suffix(dscfg.FORECASTED_COLUMN_NAME, dscfg.RISK_RATE_FACTOR_VALUE_COLUMN_NAME_SUFFIX)
        return dataset


class CreateResultsOutputDatasetStepInput(StepInput):
    """Input for CreateResultsOutputDatasetStep"""

    input_dataset: pd.DataFrame
    predicted_dataset: pd.DataFrame
    predicted_risk_rates_dataset: pd.DataFrame
    statistical_risk_rates_dataset: pd.DataFrame

    def __init__(self,
                 input_dataset: pd.DataFrame,
                 predicted_dataset: pd.DataFrame,
                 predicted_risk_rates_dataset: pd.DataFrame,
                 statistical_risk_rates_dataset: pd.DataFrame):
        """
        Class constructor
        :param input_dataset: the initial and prepared dataset
        :param predicted_dataset: the dataset with predicted probabilities
        :param predicted_risk_rates_dataset: the dataset with risk rates for predicted dataset, given the thresholds
        :param statistical_risk_rates_dataset: the dataset with statistical risk rates for dataset
        """
        self.input_dataset = input_dataset
        self.predicted_dataset = predicted_dataset
        self.predicted_risk_rates_dataset = predicted_risk_rates_dataset
        self.statistical_risk_rates_dataset = statistical_risk_rates_dataset


class CreateResultsOutputDatasetStepOutput(StepOutput):
    """Output for CreateResultsOutputDatasetStep"""

    output_dataset: pd.DataFrame

    def __init__(self, output_dataset: pd.DataFrame):
        """
        Class constructor
        :param output_dataset: the output dataset
        """
        self.output_dataset = output_dataset
