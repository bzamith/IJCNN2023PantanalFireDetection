"""
Module which contains the Scaler class
It contains the required methods to perform scaling (normalization)
"""
import copy
from typing import Any, List

import pandas as pd

import config.dataset_settings as dscfg

from src.enum.scaling_methods_enum import ScalingMethodEnum


class Scaler:
    """The Scaler entity"""

    method: ScalingMethodEnum
    base_scaler: Any
    scaler: Any

    def fit_scale(self, data: pd.DataFrame, columns: List[str] = dscfg.COLUMNS_SCALING) -> pd.DataFrame:
        """
        Fits and scales the data
        :param data: the data to be fitted and scaled
        :param columns: the columns to use for normalizing
        :return: the new dataframe
        """
        if self.__class__ == Scaler:
            raise Exception("Class Scaler must not be called directly")

        data_output = data.copy()
        self.scaler = copy.deepcopy(self.base_scaler)
        data_output[columns] = self.scaler.fit_transform(data_output[columns])
        return data_output

    def scale(self, data: pd.DataFrame, columns: List[str] = dscfg.COLUMNS_SCALING) -> pd.DataFrame:
        """
        Normalizes the data
        :param data: the data to be scaled
        :param columns: the columns to use for normalizing
        :return: the new dataframe
        """
        if self.__class__ == Scaler:
            raise Exception("Class Scaler must not be called directly")
        if self.scaler is None:
            raise Exception("You must train the scaler before calling scale method")

        data_output = data.copy()
        data_output[columns] = self.scaler.transform(data_output[columns])
        return data_output

    def descale(self, data: pd.DataFrame, columns: List[str] = dscfg.COLUMNS_SCALING) -> pd.DataFrame:
        """
        Descales the data
        :param data: the data to be descaled
        :param columns: the columns to use for denormalizing
        :return: the new dataframe
        """
        if self.__class__ == Scaler:
            raise Exception("Class Scaler must not be called directly")
        if self.scaler is None:
            raise Exception("You must train the scaler before calling descale method")

        data_output = data.copy()
        data_output[columns] = self.scaler.inverse_transform(data_output[columns])
        return data_output
