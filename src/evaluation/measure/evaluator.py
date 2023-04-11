"""
Module which contains the Evaluator class
It contains the required methods to evaluation the results based on a given measure
"""

import pandas as pd

import config.dataset_settings as dscfg

from src.enum.evaluation_measures_enum import EvaluationMeasureEnum


class Evaluator:
    """The Evaluator entity"""

    evaluation_measure: EvaluationMeasureEnum

    def calculate(self, dataset: pd.DataFrame) -> dict:
        """
        Calculates evaluation measure
        :param dataset: The dataset for which the measures will be calculated
        :return: A dict with values for each measure
        """
        if self.__class__ == Evaluator:
            raise Exception("Class Evaluator must not be called directly")
        if dataset.empty:
            raise ValueError("Parameter dataset must not be empty")
        return {}

    def format_output(self, measures_dict: dict) -> pd.DataFrame:
        """
        Formats the output and transforms into Dataframe
        :param measures_dict: The dict containing the measure values
        :return: The dataframe with the results of evaluator
        """
        output = {dscfg.EVALUATION_METRIC_COLUMN_NAME: [self.evaluation_measure.value]}
        for key in measures_dict:
            output[key] = [measures_dict[key]]
        return pd.DataFrame(output)
