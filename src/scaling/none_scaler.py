"""
Module which contains the NoneScaler class
It contains the required methods to scale according to NoneScaler
"""
from typing import List

import pandas as pd

import config.dataset_settings as dscfg

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.scaling.scaler import Scaler


class NoneScaler(Scaler):
    """The NoneScaler entity"""

    method = ScalingMethodEnum.NONE
    base_scaler = None
    scaler = None

    def fit_scale(self,
                      data: pd.DataFrame,
                      columns: List[str] = dscfg.COLUMNS_SCALING) -> pd.DataFrame:
        """
        No normalization is performed
        :param data: the data to be fitted and scaled
        :param columns: the columns to use for normalizing
        :return: the new dataframe
        """
        return data

    def scale(self,
                  data: pd.DataFrame,
                  columns: List[str] = dscfg.COLUMNS_SCALING) -> pd.DataFrame:
        """
        No normalization is performed
        :param data: the data to be scaled
        :param columns: the columns to use for normalizing
        :return: the new dataframe
        """
        return data

    def descale(self,
                    data: pd.DataFrame,
                    columns: List[str] = dscfg.COLUMNS_SCALING) -> pd.DataFrame:
        """
        No denormalization is performed
        :param data: the data to be descaled
        :param columns: the columns to use for denormalizing
        :return: the new dataframe
        """
        return data
