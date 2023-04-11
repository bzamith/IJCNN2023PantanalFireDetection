"""
Module which contains the StandardScaler class
It contains the required methods to scale according to StandardScaler
"""
from sklearn import preprocessing

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.scaling.scaler import Scaler


class StandardScaler(Scaler):
    """The StandardScaler entity"""

    method = ScalingMethodEnum.STANDARD_SCALER
    base_scaler = preprocessing.StandardScaler()
    scaler = None
