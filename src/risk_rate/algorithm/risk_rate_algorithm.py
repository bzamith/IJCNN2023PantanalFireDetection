"""
Module which contains the RiskRateAlgorithm class
It contains the required methods to calculate risk rate for a given algorithm
"""

from typing import List

import pandas as pd

import config.dataset_settings as dscfg


class RiskRateAlgorithm:
    """The RiskRateAlgorithm entity"""

    risk_rate_algorithm = None
    value_list: List[float]
    index_list = List[str]

    def calculate_for_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Risk Rate given a dataset
        :param dataset: the input dataset
        :return: dataset with risk rates
        """
        if self.__class__ == RiskRateAlgorithm:
            raise Exception("Class RiskRateAlgorithm must not be called directly")
        if dataset.empty:
            raise ValueError("Parameter dataset must not be empty")
        self.value_list = []
        self.index_list = []
        return pd.DataFrame()

    def create_risk_rate_dataset(self) -> pd.DataFrame:
        """
        Formats and creates the risk rate dataset
        :return: formatted dataset with risk rates
        """
        index_column_name = self.risk_rate_algorithm.value + dscfg.COMMON_SEPARATOR + dscfg.RISK_RATE_VALUE_COLUMN_NAME_SUFFIX
        classification_column_name = self.risk_rate_algorithm.value + dscfg.COMMON_SEPARATOR + dscfg.RISK_RATE_INDEX_COLUMN_NAME_SUFFIX
        return pd.DataFrame({
            index_column_name: self.value_list,
            classification_column_name: self.index_list
        })
