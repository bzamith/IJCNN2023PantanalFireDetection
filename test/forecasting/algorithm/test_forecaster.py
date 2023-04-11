import numpy as np
import pandas as pd
import pytest

import config.forecast_settings as fccfg

from src.forecasting.algorithm.forecaster import Forecaster


def test_constructor_call_directly_build_architecture():
    dataset = pd.DataFrame({'col': ['value']})
    with pytest.raises(Exception) as e_info:
        Forecaster().build_architecture(dataset)
    assert str(
        e_info.value) == "Class Forecaster must not be called directly"


def test_constructor_call_directly_learn():
    dataset = pd.DataFrame({'col': ['value']})
    with pytest.raises(Exception) as e_info:
        Forecaster().learn(dataset)
    assert str(
        e_info.value) == "Class Forecaster must not be called directly"


def test_constructor_call_directly_learn():
    dataset = pd.DataFrame({'col': ['value']})
    with pytest.raises(Exception) as e_info:
        Forecaster().learn(dataset)
    assert str(
        e_info.value) == "Class Forecaster must not be called directly"


def test_constructor_call_directly_forecast():
    dataset = pd.DataFrame({'col': ['value']})
    with pytest.raises(Exception) as e_info:
        Forecaster().forecast(dataset)
    assert str(
        e_info.value) == "Class Forecaster must not be called directly"


def test_get_assets():
    actual_observation_window = fccfg.OBSERVATION_WINDOW
    fccfg.OBSERVATION_WINDOW = 7

    dataset = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'col2': [10, 11, 12, 13, 14, 15, 16, 17, 18]
    })

    expected_X_0 = np.asarray(pd.DataFrame({
        'col1': [1, 2, 3, 4, 5, 6, 7],
        'col2': [10, 11, 12, 13, 14, 15, 16]
    }))

    expected_X_1 = np.asarray(pd.DataFrame({
        'col1': [2, 3, 4, 5, 6, 7, 8],
        'col2': [11, 12, 13, 14, 15, 16, 17]
    }))

    expected_y_0 = np.asarray([8, 17])
    expected_y_1 = np.asarray([9, 18])

    expected_n_features = dataset.shape[1]

    actual_X, actual_y, actual_n_features = Forecaster().get_assets(dataset)

    assert np.array_equal(actual_X[0], expected_X_0)
    assert np.array_equal(actual_X[1], expected_X_1)
    assert np.array_equal(actual_y[0], expected_y_0)
    assert np.array_equal(actual_y[1], expected_y_1)
    assert actual_n_features == expected_n_features

    fccfg.OBSERVATION_WINDOW = actual_observation_window
