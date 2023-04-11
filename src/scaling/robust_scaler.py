"""
Module which contains the RobustScaler class
It contains the required methods to scale according to RobustScaler
"""
from sklearn import preprocessing

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.scaling.scaler import Scaler


class RobustScaler(Scaler):
    """The RobustScaler entity"""

    method = ScalingMethodEnum.ROBUST_SCALER
    base_scaler = preprocessing.RobustScaler()
    scaler = None
