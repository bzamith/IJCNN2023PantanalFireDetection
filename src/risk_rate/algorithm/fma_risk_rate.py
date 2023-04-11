"""
Module which contains the FMARiskRate class
It contains the required methods to calculate the FMA risk rate values and indexes
"""

import pandas as pd

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils
from src.enum.risk_rate_algorithms_enum import RiskRateAlgorithmEnum
from src.enum.risk_rate_index_enum import RiskRateIndexEnum
from src.risk_rate.algorithm.risk_rate_algorithm import RiskRateAlgorithm


class FMARiskRate(RiskRateAlgorithm):
    """The FMARiskRate entity"""

    risk_rate_algorithm = RiskRateAlgorithmEnum.FMA

    def calculate_for_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates FMA risk rates given a dataset
        :param dataset: the input dataset
        :return: dataset with FMA risk rates
        """
        super(FMARiskRate, self).calculate_for_dataset(dataset)
        previous_fma_index = 0
        for index, row in dataset.iterrows():
            rh = pdutils.select_value_row_column(dataset, index, dscfg.RELATIVE_HUMIDITY_COLUMN_NAME)
            precip = pdutils.select_value_row_column(dataset, index, dscfg.PRECIPITATION_COLUMN_NAME)
            fma_index = self.__calculate_fma_index(previous_fma_index, rh, precip)
            self.value_list.append(fma_index)
            previous_fma_index = fma_index
            self.index_list.append(self.__calculate_fma_risk(fma_index))
        return self.create_risk_rate_dataset()

    def __calculate_fma_index(self,
                              previous_fma_index: float,
                              rh: float,
                              precip: float) -> float:
        """
        Calculates index given climatic data
        :param previous_fma_index: the FMA index from previous day
        :param rh: the relative humidity
        :param precip: the precipitation
        :return: the FMA index
        """
        factor = 100.0 / rh
        if precip > 12.90:
            fma_index = 0.00
        elif precip >= 10.00:
            fma_index = previous_fma_index * 0.20 + factor
        elif precip >= 5.00:
            fma_index = previous_fma_index * 0.40 + factor
        elif precip >= 2.50:
            fma_index = previous_fma_index * 0.70 + factor
        else:
            fma_index = previous_fma_index + factor
        return fma_index

    def __calculate_fma_risk(self,
                             fma_index: float) -> str:
        """
        Convert FMA index to risk rate
        :param fma_index: the FMA index
        :return: the risk rate
        """
        if fma_index > 20:
            return RiskRateIndexEnum.MUITO_ALTO.value
        if fma_index >= 8.1:
            return RiskRateIndexEnum.ALTO.value
        if fma_index >= 3.1:
            return RiskRateIndexEnum.MEDIO.value
        if fma_index >= 1.1:
            return RiskRateIndexEnum.PEQUENO.value
        return RiskRateIndexEnum.NULO.value
