from unittest.mock import MagicMock

import pytest

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.scaling import scaler_factory
from src.scaling.max_abs_scaler import MaxAbsScaler
from src.scaling.min_max_scaler import MinMaxScaler
from src.scaling.none_scaler import NoneScaler
from src.scaling.robust_scaler import RobustScaler
from src.scaling.standard_scaler import StandardScaler


def test_get_min_max():
    algorithm = scaler_factory.get(
        ScalingMethodEnum.MIN_MAX_SCALER)
    assert isinstance(algorithm, MinMaxScaler)


def test_get_standard():
    algorithm = scaler_factory.get(
        ScalingMethodEnum.STANDARD_SCALER)
    assert isinstance(algorithm, StandardScaler)


def test_get_max_abs():
    algorithm = scaler_factory.get(
        ScalingMethodEnum.MAX_ABS_SCALER)
    assert isinstance(algorithm, MaxAbsScaler)


def test_get_robust():
    algorithm = scaler_factory.get(
        ScalingMethodEnum.ROBUST_SCALER)
    assert isinstance(algorithm, RobustScaler)


def test_get_none():
    algorithm = scaler_factory.get(
        ScalingMethodEnum.NONE)
    assert isinstance(algorithm, NoneScaler)


def test_get_invalid():
    with pytest.raises(TypeError) as e_info:
        scaler_factory.get("xxx")
    assert str(
        e_info.value) == "Parameter scaling_method must be of type " \
                         "ScalingMethodEnum"


def test_get_null():
    with pytest.raises(ValueError) as e_info:
        scaler_factory.get(None)
    assert str(e_info.value) == "Parameter scaling_method must not be null"


def test_get_not_implemented():
    mock = MagicMock(spec=ScalingMethodEnum, name="Dummy", value="DummyValue")
    with pytest.raises(NotImplementedException) as e_info:
        scaler_factory.get(mock)
    assert str(e_info.value) == "No ScalingMethodEnum implemented for scaling algorithm DummyValue"
