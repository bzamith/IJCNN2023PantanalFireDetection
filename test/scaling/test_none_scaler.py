from unittest import mock

import pandas as pd
import pytest

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.scaling.none_scaler import NoneScaler

DATA = pd.DataFrame({'col': ['value0', 'value1'],
                     'col1': ['value2', 'value3']})
COLUMNS = ['col1']


def test_attributes():
    scaler = NoneScaler()
    assert scaler.method == ScalingMethodEnum.NONE
    assert scaler.base_scaler is None
    assert scaler.scaler is None


def test_fit_scale():
    actual_output = NoneScaler().fit_scale(DATA, COLUMNS)
    assert actual_output.equals(DATA)


def test_scale():
    actual_output = NoneScaler().scale(DATA, COLUMNS)
    assert actual_output.equals(DATA)


def test_descale():
    actual_output = NoneScaler().descale(DATA, COLUMNS)
    assert actual_output.equals(DATA)

