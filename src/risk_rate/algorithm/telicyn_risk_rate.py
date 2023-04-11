"""
Module which contains the TelicynRiskRate class
It contains the required methods to calculate the Telicyn risk rate values and indexes
"""

from math import log10

import pandas as pd

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils
from src.enum.risk_rate_algorithms_enum import RiskRateAlgorithmEnum
from src.enum.risk_rate_index_enum import RiskRateIndexEnum
from src.risk_rate.algorithm.risk_rate_algorithm import RiskRateAlgorithm


class TelicynRiskRate(RiskRateAlgorithm):
    """The TelicynRiskRate entity"""

    risk_rate_algorithm = RiskRateAlgorithmEnum.TELICYN

    def calculate_for_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Telicyn risk rates given a dataset
        :param dataset: the input dataset
        :return: dataset with Telicyn risk rates
        """
        super(TelicynRiskRate, self).calculate_for_dataset(dataset)
        previous_telicyn_index = 0
        for index, row in dataset.iterrows():
            temp = pdutils.select_value_row_column(dataset, index, dscfg.TEMPERATURE_COLUMN_NAME)
            rh = pdutils.select_value_row_column(dataset, index, dscfg.RELATIVE_HUMIDITY_COLUMN_NAME)
            precip = pdutils.select_value_row_column(dataset, index, dscfg.PRECIPITATION_COLUMN_NAME)
            es = 6.1 * pow(10, ((7.5 * temp) / (237.3 + temp)))
            e = (rh / 100) * es
            log_dpt = log10(e / 6.1)
            dpt = (237.3 * log_dpt) / (7.5 - log_dpt)

            telicyn_index = self.__calculate_telicyn_index(previous_telicyn_index, temp, dpt, precip)
            self.value_list.append(telicyn_index)
            previous_telicyn_index = telicyn_index
            self.index_list.append(self.__calculate_telicyn_risk(telicyn_index))
        return self.create_risk_rate_dataset()

    def __calculate_telicyn_index(self,
                                  previous_telicyn_index: float,
                                  temp: float,
                                  dpt: float,
                                  precip: float) -> float:
        """
        Calculates index given climatic data
        :param previous_telicyn_index: the Telicyn index from previous day
        :param temp: the temperature
        :param dpt: dew point temperature
        :param precip: the precipitation
        :return: the FMA index
        """
        diff = temp - dpt
        factor = 0 if diff <= 0 else log10(diff)
        if precip < 2.5:
            telicyn_index = previous_telicyn_index + factor
        else:
            telicyn_index = 0
        return telicyn_index

    def __calculate_telicyn_risk(self,
                                 telicyn_index: float) -> str:
        """
        Convert Telicyn index to risk rate
        :param telicyn_index: the Telicyn index
        :return: the risk rate
        """
        if telicyn_index > 5:
            return RiskRateIndexEnum.ALTO.value
        if telicyn_index >= 3.6:
            return RiskRateIndexEnum.MEDIO.value
        if telicyn_index >= 2.1:
            return RiskRateIndexEnum.PEQUENO.value
        return RiskRateIndexEnum.NULO.value
