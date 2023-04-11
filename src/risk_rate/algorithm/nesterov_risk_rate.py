"""
Module which contains the NesterovRiskRate class
It contains the required methods to calculate the Nesterov risk rate values and indexes
"""

import pandas as pd

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils
from src.enum.risk_rate_algorithms_enum import RiskRateAlgorithmEnum
from src.enum.risk_rate_index_enum import RiskRateIndexEnum
from src.risk_rate.algorithm.risk_rate_algorithm import RiskRateAlgorithm


class NesterovRiskRate(RiskRateAlgorithm):
    """The NesterovRiskRate entity"""

    risk_rate_algorithm = RiskRateAlgorithmEnum.NESTEROV

    def calculate_for_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Nesterov risk rates given a dataset
        :param dataset: the input dataset
        :return: dataset with Nesterov risk rates
        """
        super(NesterovRiskRate, self).calculate_for_dataset(dataset)
        previous_nesterov_index = 0
        for index, row in dataset.iterrows():
            temp = pdutils.select_value_row_column(dataset, index, dscfg.TEMPERATURE_COLUMN_NAME)
            rh = pdutils.select_value_row_column(dataset, index, dscfg.RELATIVE_HUMIDITY_COLUMN_NAME)
            precip = pdutils.select_value_row_column(dataset, index, dscfg.PRECIPITATION_COLUMN_NAME)

            es = 6.1 * pow(10, ((7.5 * temp) / (237.3 + temp)))
            d = es * (1 - (rh / 100))

            nesterov_index = self.__calculate_nesterov_index(previous_nesterov_index, temp, d, precip)
            self.value_list.append(nesterov_index)
            previous_nesterov_index = nesterov_index
            self.index_list.append(self.__calculate_nesterov_risk(nesterov_index))
        return self.create_risk_rate_dataset()

    def __calculate_nesterov_index(self,
                                   previous_nesterov_index: float,
                                   temp: float,
                                   d: float,
                                   precip: float) -> float:
        """
        Calculates index given climatic data
        :param previous_nesterov_index: the Nesterov index from previous day
        :param temp: the temperature
        :param d: saturation deficit
        :param precip: the precipitation
        :return: the Nesterov index
        """
        factor = d * temp
        if precip > 10.00:
            nesterov_index = 0
        elif precip >= 8.10:
            nesterov_index = factor
        elif precip >= 5.10:
            nesterov_index = previous_nesterov_index * 0.50 + factor
        elif precip >= 2.10:
            nesterov_index = previous_nesterov_index * 0.75 + factor
        else:
            nesterov_index = previous_nesterov_index + factor
        return nesterov_index

    def __calculate_nesterov_risk(self,
                                  nesterov_index: float) -> str:
        """
        Convert Nesterov index to risk rate
        :param nesterov_index: the Nesterov index
        :return: the risk rate
        """
        if nesterov_index > 4000:
            return RiskRateIndexEnum.MUITO_ALTO.value
        if nesterov_index >= 1001:
            return RiskRateIndexEnum.ALTO.value
        if nesterov_index >= 501:
            return RiskRateIndexEnum.MEDIO.value
        if nesterov_index >= 301:
            return RiskRateIndexEnum.PEQUENO.value
        return RiskRateIndexEnum.NULO.value
