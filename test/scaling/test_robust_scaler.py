from unittest import mock

import sklearn.preprocessing as pp

import pandas as pd
import pytest

from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.scaling.robust_scaler import RobustScaler

DATA = pd.DataFrame({'col': ['value0', 'value1'],
                     'col1': ['value2', 'value3']})
COLUMNS = ['col1']


def test_attributes():
    scaler = RobustScaler()
    assert scaler.method == ScalingMethodEnum.ROBUST_SCALER
    assert isinstance(scaler.base_scaler, pp.RobustScaler)
    assert scaler.scaler is None


@mock.patch('sklearn.preprocessing.RobustScaler.fit_transform')
def test_fit_scale(mock_scaler_fit_transform):
    return_scaler = pd.DataFrame({'col1': ['value4', 'value5']})
    expected_output = pd.DataFrame({'col': ['value0', 'value1'],
                                    'col1': ['value4', 'value5']})

    mock_scaler_fit_transform.return_value = return_scaler

    actual_output = RobustScaler().fit_scale(DATA, COLUMNS)

    mock_scaler_fit_transform.assert_called_once()
    assert actual_output.equals(expected_output)


@mock.patch('sklearn.preprocessing.RobustScaler.transform')
def test_scale(mock_scaler_fit_transform):
    return_scaler = pd.DataFrame({'col1': ['value4', 'value5']})
    expected_output = pd.DataFrame({'col': ['value0', 'value1'],
                                    'col1': ['value4', 'value5']})

    mock_scaler_fit_transform.return_value = return_scaler

    scaler = RobustScaler()
    scaler.scaler = scaler.base_scaler
    actual_output = scaler.scale(DATA, COLUMNS)

    mock_scaler_fit_transform.assert_called_once()
    assert actual_output.equals(expected_output)


def test_scale_none_scaler():
    with pytest.raises(Exception) as e_info:
        RobustScaler().scale(DATA, COLUMNS)
    assert str(e_info.value) == "You must train the scaler before calling scale method"


@mock.patch('sklearn.preprocessing.RobustScaler.inverse_transform')
def test_descale(mock_scaler_inverse_transform):
    return_scaler = pd.DataFrame({'col1': ['value4', 'value5']})
    expected_output = pd.DataFrame({'col': ['value0', 'value1'],
                                    'col1': ['value4', 'value5']})

    mock_scaler_inverse_transform.return_value = return_scaler

    scaler = RobustScaler()
    scaler.scaler = scaler.base_scaler
    actual_output = scaler.descale(DATA, COLUMNS)

    mock_scaler_inverse_transform.assert_called_once()
    assert actual_output.equals(expected_output)


def test_constructor_descale_empty_columns():
    with pytest.raises(Exception) as e_info:
        RobustScaler().descale(DATA, COLUMNS)
    assert str(e_info.value) == "You must train the scaler before calling descale method"
