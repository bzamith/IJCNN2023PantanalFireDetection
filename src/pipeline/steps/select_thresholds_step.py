"""
Module which contains the SelectThresholdsStep, SelectThresholdsStepInput and SelectThresholdsStepOutput classes
They contain the required methods to calculate the risk rates after prediction (classification algorithm)
"""

from typing import Any, List, Tuple

import pandas as pd

import config.dataset_settings as dscfg
import config.general_settings as cfg
import config.risk_rate_threshold_settings as rrtcfg

import src.utils.pandas_utils as pdutils
from src.enum.evaluation_measures_enum import EvaluationMeasureEnum
from src.enum.risk_rate_index_enum import RiskRateIndexEnum
from src.evaluation.measure.evaluator_factory import get
from src.pipeline.step import Step, StepInput, StepOutput
from src.utils.dataset_columns_utils import get_column_name_prefix, get_column_name_suffix, get_column_names_suffix


class SelectThresholdsStep(Step):
    """The SelectThresholdsStep entity"""

    step_name = "Calculate and Select Predicted Risk Rates"
    step_description = "Calculate the risk rates for the predicted results, and selects the thresholds"

    def __init__(self, original_dataset: pd.DataFrame,
                 predicted_dataset: pd.DataFrame):
        """
        Class constructor
        :param original_dataset: the dataset with original data
        :param predicted_dataset: the dataset with predicted probabilities
        """
        self.step_input = SelectThresholdsStepInput(original_dataset, predicted_dataset)
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        original_dataset = self.step_input.original_dataset
        predicted_dataset = self.step_input.predicted_dataset

        dates = pdutils.select_columns(predicted_dataset, dscfg.DATE_COLUMN_NAME, reset_row_indexes=True)
        merged_dataframe = pdutils.join_inner_by_date([dates, original_dataset])
        hotspot_identified = pdutils.select_columns(merged_dataframe, dscfg.HOTSPOT_IDENTIFIED_COLUMN_NAME, reset_row_indexes=True)

        best_risk_rates_dataset = None
        best_correlation_dataset = None
        best_score = -1
        best_thresholds = []

        for thr_nulo in rrtcfg.THRESHOLD_NULO:
            for thr_pequeno in rrtcfg.THRESHOLD_PEQUENO:
                for thr_medio in rrtcfg.THRESHOLD_MEDIO:
                    for thr_alto in rrtcfg.THRESHOLD_ALTO:
                        for thr_muito_alto in rrtcfg.THRESHOLD_MUITO_ALTO:
                            curr_thresholds = [thr_nulo, thr_pequeno, thr_medio, thr_alto, thr_muito_alto]
                            if self.__are_valid_thresholds(curr_thresholds):
                                RiskRateIndexEnum.NULO.set_prob_threshold(thr_nulo)
                                RiskRateIndexEnum.PEQUENO.set_prob_threshold(thr_pequeno)
                                RiskRateIndexEnum.MEDIO.set_prob_threshold(thr_medio)
                                RiskRateIndexEnum.ALTO.set_prob_threshold(thr_alto)
                                RiskRateIndexEnum.MUITO_ALTO.set_prob_threshold(thr_muito_alto)

                                probs_present, risk_rates_present = self.__get_risk_rates_present(predicted_dataset)
                                probs_forecasted_columns, risk_rates_forecasted_list = self.__get_risk_rates_future(predicted_dataset)
                                risk_rates_dataset = self.__create_risk_rate_dataset(
                                    dates, probs_present, risk_rates_present,
                                    probs_forecasted_columns, risk_rates_forecasted_list,
                                    hotspot_identified
                                )
                                correlation_dataset = self.__create_correlation_dataset(risk_rates_dataset)
                                score = self.__get_score(correlation_dataset)
                                if score > best_score:
                                    best_score = score
                                    best_risk_rates_dataset = risk_rates_dataset
                                    best_correlation_dataset = correlation_dataset
                                    best_thresholds = curr_thresholds

        self.__save_selected_thresholds_info(best_thresholds)

        self.step_output = SelectThresholdsStepOutput(best_risk_rates_dataset, best_correlation_dataset, best_score, best_thresholds)

    def __are_valid_thresholds(self, thresholds: List[float]) -> bool:
        """
        Returns if thresholds are valid or not
        :param thresholds: the list with the thresholds
        :return: whether thresholds are valid or not
        """
        for i in range(0, len(thresholds)):
            for j in range(0, len(thresholds)):
                if thresholds[i] >= thresholds[j] and i < j:
                    return False
        return True

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

        return pdutils.join_dataframes_y_wise([dates, pd.DataFrame(data_map), hotspot_identified])

    def __create_correlation_dataset(self, risk_rate_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Gets the correlation score rates for dataset
        :param risk_rate_dataset: the input dataset with risk rates
        :return: the correlation
        """
        evaluation_calculator = get(EvaluationMeasureEnum.CORRELATION)
        result = evaluation_calculator.calculate(risk_rate_dataset)
        return pd.DataFrame(result)

    def __get_score(self, correlation_dataset: pd.DataFrame) -> float:
        """
        Get score for a given combination
        :param correlation_dataset: the dataset with the correlations
        :return: the score
        """
        total_sum = 0

        for column in list(correlation_dataset.columns):
            is_present_result_column = dscfg.PRESENT_COLUMN_NAME in column
            is_forecasted_result_column = dscfg.FORECASTED_COLUMN_NAME in column

            curr_sum = 0

            if (is_present_result_column or is_forecasted_result_column) and dscfg.COUNT_SUFFIX not in column:
                nulo_corr = pdutils.select_value_row_column(correlation_dataset, 0, column)
                pequeno_corr = pdutils.select_value_row_column(correlation_dataset, 1, column)
                medio_corr = pdutils.select_value_row_column(correlation_dataset, 2, column)
                alto_corr = pdutils.select_value_row_column(correlation_dataset, 3, column)
                muito_alto_corr = pdutils.select_value_row_column(correlation_dataset, 4, column)

                curr_sum += (1 - nulo_corr) * 5
                curr_sum += (1 - pequeno_corr) * 4
                curr_sum += (1 - medio_corr) * 3
                curr_sum += alto_corr * 2
                curr_sum += muito_alto_corr * 1

                curr_sum = curr_sum / 15

                total_sum += curr_sum

        return total_sum

    def __save_selected_thresholds_info(self, selected_thresholds: List[float]) -> None:
        with open(cfg.ASSETS_DIR + "thresholds.txt", 'w') as f:
            f.write("Selected best thresholds: ")
            f.write("\n")
            f.write(','.join(str(x) for x in selected_thresholds))


class SelectThresholdsStepInput(StepInput):
    """Input for SelectThresholdsStep"""

    original_dataset: pd.DataFrame
    predicted_dataset: pd.DataFrame

    def __init__(self, original_dataset: pd.DataFrame,
                 predicted_dataset: pd.DataFrame):
        """
        Class constructor
        :param original_dataset: the dataset with original data
        :param predicted_dataset: the dataset with predicted probabilities
        """
        self.original_dataset = original_dataset
        self.predicted_dataset = predicted_dataset


class SelectThresholdsStepOutput(StepOutput):
    """Output for SelectThresholdsStep"""

    predicted_risk_rates_dataset: pd.DataFrame
    correlation_dataset: pd.DataFrame
    score: float
    thresholds: List[float]

    def __init__(self, predicted_risk_rates_dataset: pd.DataFrame,
                 correlation_dataset: pd.DataFrame,
                 score: float,
                 thresholds: List[float]):
        """
        Class constructor
        :param predicted_risk_rates_dataset: the dataset with risk rates for predicted dataset, given the thresholds
        :param correlation_dataset: the dataset with correlations for predicted dataset, given the thresholds
        :param score: the best score, given the thresholds
        :param thresholds: the thresholds obtained for the best score
        """
        self.predicted_risk_rates_dataset = predicted_risk_rates_dataset
        self.correlation_dataset = correlation_dataset
        self.score = score
        self.thresholds = thresholds
