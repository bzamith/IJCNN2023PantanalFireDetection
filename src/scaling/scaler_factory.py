"""Module which represents a factory for Scaler"""
from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.scaling.max_abs_scaler import MaxAbsScaler
from src.scaling.min_max_scaler import MinMaxScaler
from src.scaling.none_scaler import NoneScaler
from src.scaling.robust_scaler import RobustScaler
from src.scaling.scaler import Scaler
from src.scaling.standard_scaler import StandardScaler


def get(scaling_method: ScalingMethodEnum) -> Scaler:
    """
    Factory method for ScalingMethods
    :param scaling_method: the scaling method
    :return: the scaler for that scaling method
    """
    if not scaling_method:
        raise ValueError("Parameter scaling_method must not be null")
    if not isinstance(scaling_method, ScalingMethodEnum):
        raise TypeError("Parameter scaling_method must be of type ScalingMethodEnum")
    if scaling_method == ScalingMethodEnum.NONE:
        return NoneScaler()
    if scaling_method == ScalingMethodEnum.MIN_MAX_SCALER:
        return MinMaxScaler()
    if scaling_method == ScalingMethodEnum.STANDARD_SCALER:
        return StandardScaler()
    if scaling_method == ScalingMethodEnum.ROBUST_SCALER:
        return RobustScaler()
    if scaling_method == ScalingMethodEnum.MAX_ABS_SCALER:
        return MaxAbsScaler()
    raise NotImplementedException("No ScalingMethodEnum implemented for scaling algorithm {}".format(scaling_method.value))
