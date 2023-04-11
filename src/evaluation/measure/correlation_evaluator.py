"""
Module which contains the CorrelationEvaluator class
It contains the required methods to calculate the evaluation measure
based on the correlations between hotspot identified and risk rate index
"""
import pandas as pd

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils
from src.enum.evaluation_measures_enum import EvaluationMeasureEnum
from src.enum.risk_rate_index_enum import RiskRateIndexEnum
from src.evaluation.measure.evaluator import Evaluator


class CorrelationEvaluator(Evaluator):
    """The CorrelationEvaluator entity"""

    evaluation_measure = EvaluationMeasureEnum.CORRELATION

    def calculate(self, dataset: pd.DataFrame) -> dict:
        """
        Calculates evaluation measure
        :param dataset: the dataset for which the measures will be calculated
        :return: a dict with values for each correlation
        """
        super(CorrelationEvaluator, self).calculate(dataset)
        index_suffix = dscfg.RISK_RATE_INDEX_COLUMN_NAME_SUFFIX

        shrink_dataset = pdutils.select_columns_with_substring(dataset, index_suffix)

        column_names = list(shrink_dataset.columns)
        output = {}

        measure_names = []
        for risk_rate in RiskRateIndexEnum:
            measure_name = self.evaluation_measure.value + dscfg.COMMON_SEPARATOR + risk_rate.value
            measure_names.append(measure_name)
        output[dscfg.EVALUATION_METRIC_COLUMN_NAME] = measure_names

        for column_name in column_names:
            correlation = []
            count = []
            prefix = column_name[0:(len(column_name) - len(index_suffix) - 1)]
            for risk_rate in RiskRateIndexEnum:
                curr_correlation = self.__calculate_correlation_rate(dataset, column_name, risk_rate.value)
                curr_count = dataset[dataset[column_name] == risk_rate.value].shape[0]
                correlation.append(curr_correlation)
                count.append(curr_count)
            output[prefix] = correlation
            output[prefix + dscfg.COMMON_SEPARATOR + dscfg.COUNT_SUFFIX] = count
        return output

    def __calculate_correlation_rate(self,
                                     data: pd.DataFrame,
                                     column_name: str, rate: str) -> float:
        """
        Calculates correlation rate
        :param data: the data for which the measures will be calculated
        :param column_name: the column to be considered for calculation
        :return: correlation rate for column in data
        """
        rate_data = pdutils.select_rows_by_value(data, column_name, rate)
        if rate_data.empty:
            return 0
        rate_data_true = pdutils.select_rows_by_value(rate_data,
                                                      dscfg.HOTSPOT_IDENTIFIED_COLUMN_NAME,
                                                      1)
        return rate_data_true.shape[0] / rate_data.shape[0]
