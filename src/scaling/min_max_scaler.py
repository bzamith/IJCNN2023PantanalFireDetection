"""
Module which contains the MinMaxScaler class
It contains the required methods to scale according to MinMaxScaler
"""
from sklearn import preprocessing

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.scaling.scaler import Scaler


class MinMaxScaler(Scaler):
    """The MinMaxScaler entity"""

    method = ScalingMethodEnum.MIN_MAX_SCALER
    base_scaler = preprocessing.MinMaxScaler()
    scaler = None
