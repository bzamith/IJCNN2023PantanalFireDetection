"""
Module which contains the MaxAbsScaler class
It contains the required methods to scale according to MaxAbsScaler
"""
from sklearn import preprocessing

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.scaling.scaler import Scaler


class MaxAbsScaler(Scaler):
    """The MaxAbsScaler entity"""

    method = ScalingMethodEnum.MAX_ABS_SCALER
    base_scaler = preprocessing.MaxAbsScaler()
    scaler = None
