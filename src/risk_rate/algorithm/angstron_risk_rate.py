"""
Module which contains the AngstronRiskRate class
It contains the required methods to calculate the Angstron risk rate values and indexes
"""
import pandas as pd

import config.dataset_settings as dscfg

import src.utils.pandas_utils as pdutils
from src.enum.risk_rate_algorithms_enum import RiskRateAlgorithmEnum
from src.enum.risk_rate_index_enum import RiskRateIndexEnum
from src.risk_rate.algorithm.risk_rate_algorithm import RiskRateAlgorithm


class AngstronRiskRate(RiskRateAlgorithm):
    """The AngstronRiskRate entity"""

    risk_rate_algorithm = RiskRateAlgorithmEnum.ANGSTRON

    def calculate_for_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Angstron risk rates given a dataset
        :param dataset: the input dataset
        :return: dataset with angstron risk rates
        """
        super(AngstronRiskRate, self).calculate_for_dataset(dataset)
        for index, row in dataset.iterrows():
            temp = pdutils.select_value_row_column(
                dataset, index, dscfg.TEMPERATURE_COLUMN_NAME)
            rh = pdutils.select_value_row_column(
                dataset, index, dscfg.RELATIVE_HUMIDITY_COLUMN_NAME)

            angstron_index = self.__calculate_angstron_index(rh, temp)
            self.value_list.append(angstron_index)
            self.index_list.append(self.__calculate_angstron_risk(angstron_index))
        return self.create_risk_rate_dataset()

    def __calculate_angstron_index(self,
                                   rh: float,
                                   temp: float) -> float:
        """
        Calculates index given relative humidity and temperature
        :param rh: the relative humidity
        :param temp: the temperature
        :return: the angstron index
        """
        return (rh / 20) + ((temp - 27) / 10)

    def __calculate_angstron_risk(self,
                                  angstron_index: float) -> str:
        """
        Convert angstron index to risk rate
        :param angstron_index: the angstron index
        :return: the risk rate
        """
        if angstron_index > 4.5:
            return RiskRateIndexEnum.NULO.value
        if angstron_index >= 4.3:
            return RiskRateIndexEnum.PEQUENO.value
        if angstron_index >= 4.0:
            return RiskRateIndexEnum.MEDIO.value
        if angstron_index >= 3.5:
            return RiskRateIndexEnum.ALTO.value
        return RiskRateIndexEnum.MUITO_ALTO.value
