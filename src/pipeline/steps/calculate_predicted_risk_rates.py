"""
Module which contains the CalculatePredictedRiskRatesStep, CalculatePredictedRiskRatesStepInput and CalculatePredictedRiskRatesStepOutput classes
They contain the required methods to calculate the risk rates after prediction (classification algorithm)
"""

from typing import Any, List, Tuple

import pandas as pd

import config.dataset_settings as dscfg
import config.general_settings as cfg

import src.utils.pandas_utils as pdutils
from src.enum.risk_rate_index_enum import RiskRateIndexEnum
from src.pipeline.step import Step, StepInput, StepOutput
from src.utils.dataset_columns_utils import get_column_name_prefix, get_column_name_suffix, get_column_names_suffix


class CalculatePredictedRiskRatesStep(Step):
    """The CalculatePredictedRiskRatesStep entity"""

    step_name = "Calculate and Select Predicted Risk Rates"
    step_description = "Calculate the risk rates for the predicted results, and selects the thresholds"

    def __init__(self, original_dataset: pd.DataFrame,
                 predicted_dataset: pd.DataFrame,
                 thresholds: List[float]):
        """
        Class constructor
        :param original_dataset: the dataset with original data
        :param predicted_dataset: the dataset with predicted probabilities
        :param thresholds: the selected thresholds for probabilities
        """
        self.step_input = CalculatePredictedRiskRatesStepInput(original_dataset, predicted_dataset, thresholds)
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        original_dataset = self.step_input.original_dataset
        predicted_dataset = self.step_input.predicted_dataset
        thresholds = self.step_input.thresholds

        self.__set_thresholds(thresholds)

        dates = pdutils.select_columns(predicted_dataset, dscfg.DATE_COLUMN_NAME, reset_row_indexes=True)
        merged_dataframe = pdutils.join_inner_by_date([dates, original_dataset])
        try:
            hotspot_identified = pdutils.select_columns(merged_dataframe, dscfg.HOTSPOT_IDENTIFIED_COLUMN_NAME, reset_row_indexes=True)
        except KeyError:
            hotspot_identified = None

        probs_present, risk_rates_present = self.__get_risk_rates_present(predicted_dataset)
        probs_forecasted_columns, risk_rates_forecasted_list = self.__get_risk_rates_future(predicted_dataset)
        risk_rates_dataset = self.__create_risk_rate_dataset(
            dates, probs_present, risk_rates_present,
            probs_forecasted_columns, risk_rates_forecasted_list,
            hotspot_identified
        )

        risk_rates_dataset.to_csv(cfg.DATA_GENERATED_DIR + "predicted_risk_rates.csv")

        self.step_output = CalculatePredictedRiskRatesStepOutput(risk_rates_dataset)

    def __set_thresholds(self, thresholds: List[float]) -> None:
        RiskRateIndexEnum.NULO.set_prob_threshold(thresholds[0])
        RiskRateIndexEnum.PEQUENO.set_prob_threshold(thresholds[1])
        RiskRateIndexEnum.MEDIO.set_prob_threshold(thresholds[2])
        RiskRateIndexEnum.ALTO.set_prob_threshold(thresholds[3])
        RiskRateIndexEnum.MUITO_ALTO.set_prob_threshold(thresholds[4])

    def __get_risk_rates_present(self, predicted_dataset: pd.DataFrame) -> Tuple[Any, List[str]]:
        """
        Gets the risk rates for predicted dataset, present time
        :param predicted_dataset: the dataset with predicted probabilities
        :return: the risk rates
        """
        column_name = get_column_name_prefix(dscfg.PRESENT_COLUMN_NAME, dscfg.PROBS_PREFIX)
        probs_present = predicted_dataset[column_name].tolist()
        risk_rates_present = self.__generate_predicted_risk_rate(probs_present)
        return probs_present, risk_rates_present

    def __get_risk_rates_future(self, predicted_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """
        Gets the risk rates for predicted dataset, future time
        :param predicted_dataset: the dataset with predicted probabilities
        :return: the risk rates
        """
        column_name = get_column_name_prefix(dscfg.FORECASTED_COLUMN_NAME, dscfg.PROBS_PREFIX)
        probs_forecasted_columns = pdutils.select_columns_with_substring(predicted_dataset, column_name)
        risk_rates_forecasted_list = []
        for i in range(0, probs_forecasted_columns.shape[1]):
            probs_forecasted = pdutils.dataframe_to_list(pdutils.select_columns(probs_forecasted_columns, i))
            risk_rates_forecasted = self.__generate_predicted_risk_rate(probs_forecasted)
            risk_rates_forecasted_list.append(risk_rates_forecasted)
        return probs_forecasted_columns, risk_rates_forecasted_list

    def __generate_predicted_risk_rate(self, probs: List[float]) -> List[str]:
        """
        Gets the risk rates for predicted dataset
        :param probs: the list of probabilities
        :return: the risk rates
        """
        predicted_risk_rates = []
        for prob in probs:
            for risk_rate in RiskRateIndexEnum:
                if prob <= risk_rate.prob_threshold:
                    predicted_risk_rates.append(risk_rate.value)
                    break
        return predicted_risk_rates

    def __create_risk_rate_dataset(self,
                                   dates: pd.DataFrame,
                                   probs_present: List[float],
                                   risk_rates_present: List[str],
                                   probs_forecasted_columns: pd.DataFrame,
                                   risk_rates_forecasted_list: List[List[str]],
                                   hotspot_identified: pd.DataFrame) -> pd.DataFrame:
        """
        Create the dataset with risk rates
        :param dates: the dataframe with dates, to be appended
        :param probs_present: the probabilities for present
        :param risk_rates_present: the risk rates for present
        :param probs_forecasted_columns: the probabilities for future days (forecasted)
        :param risk_rates_forecasted_list: the risk rates for future days (forecasted)
        :param hotspot_identified: the dataframe with hotspot_identified (binary)
        :return: the dataset
        """
        present_value_column_name = get_column_name_suffix(dscfg.PRESENT_COLUMN_NAME, dscfg.RISK_RATE_VALUE_COLUMN_NAME_SUFFIX)
        present_index_column_name = get_column_name_suffix(dscfg.PRESENT_COLUMN_NAME, dscfg.RISK_RATE_INDEX_COLUMN_NAME_SUFFIX)
        forecasted_value_column_names = get_column_names_suffix(dscfg.FORECASTED_COLUMN_NAME, dscfg.RISK_RATE_VALUE_COLUMN_NAME_SUFFIX)
        forecasted_index_column_names = get_column_names_suffix(dscfg.FORECASTED_COLUMN_NAME, dscfg.RISK_RATE_INDEX_COLUMN_NAME_SUFFIX)

        data_map = {
            present_value_column_name: probs_present,
            present_index_column_name: risk_rates_present
        }

        for i in range(0, probs_forecasted_columns.shape[1]):
            column_name_value = forecasted_value_column_names[i]
            column_name_index = forecasted_index_column_names[i]
            data_map[column_name_value] = pdutils.dataframe_to_list(pdutils.select_columns(probs_forecasted_columns, i))
            data_map[column_name_index] = risk_rates_forecasted_list[i]

        if hotspot_identified is not None:
            return pdutils.join_dataframes_y_wise([dates, pd.DataFrame(data_map), hotspot_identified])
        return pdutils.join_dataframes_y_wise([dates, pd.DataFrame(data_map)])


class CalculatePredictedRiskRatesStepInput(StepInput):
    """Input for CalculatePredictedRiskRatesStep"""

    original_dataset: pd.DataFrame
    predicted_dataset: pd.DataFrame
    thresholds: List[float]

    def __init__(self, original_dataset: pd.DataFrame,
                 predicted_dataset: pd.DataFrame,
                 thresholds: List[float]):
        """
        Class constructor
        :param original_dataset: the dataset with original data
        :param predicted_dataset: the dataset with predicted probabilities
        :param thresholds: the selected thresholds for probabilities
        """
        self.original_dataset = original_dataset
        self.predicted_dataset = predicted_dataset
        self.thresholds = thresholds


class CalculatePredictedRiskRatesStepOutput(StepOutput):
    """Output for CalculatePredictedRiskRatesStep"""

    predicted_risk_rates_dataset: pd.DataFrame

    def __init__(self, predicted_risk_rates_dataset: pd.DataFrame):
        """
        Class constructor
        :param predicted_risk_rates_dataset: the dataset with risk rates for predicted dataset, given the thresholds
        """
        self.predicted_risk_rates_dataset = predicted_risk_rates_dataset
