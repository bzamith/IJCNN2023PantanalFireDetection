import pytest

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.utils.validation_utils import validate_enum, validate_type


def test_validate_input_enum_belongs():
    assert validate_enum("Min Max Scaler", "Scaler Algorithm", ScalingMethodEnum)


def test_validate_input_enum_does_not_belong():
    with pytest.raises(TypeError) as e_info:
        validate_enum("dummy", "Scaler Algorithm", ScalingMethodEnum)
    assert str(e_info.value) == "Parameter Scaler Algorithm must belong to ScalingMethodEnum"


def test_validate_input_type_belongs():
    assert validate_type(0, "threshold", int)


def test_validate_input_type_does_not_belong():
    with pytest.raises(TypeError) as e_info:
        validate_type("dummy", "threshold", int)
    assert str(e_info.value) == "Parameter threshold must be from type int"