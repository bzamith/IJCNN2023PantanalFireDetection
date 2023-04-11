import numpy as np
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM, MaxPooling1D, TimeDistributed

import config.forecast_settings as fccfg
from src.enum.forecasting_algorithms_enum import ForecastingAlgorithmEnum
from src.forecasting.algorithm.cnn_forecaster import CNNForecaster

import pytest


def test_attributes():
    forecaster = CNNForecaster()
    assert forecaster.algorithm == ForecastingAlgorithmEnum.CNN
    assert forecaster.forecaster is None


def test_build_architecture():
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

    actual_X, actual_y, actual_architecture = CNNForecaster().build_architecture(dataset)
    actual_architecture.compile(loss=fccfg.ERROR_METRIC, optimizer='adam', metrics=['mse', 'mae'])

    expected_X = np.zeros((2, 7, 2), dtype=int)
    expected_X[0] = expected_X_0
    expected_X[1] = expected_X_1

    expected_y = np.zeros((2, 2), dtype=int)
    expected_y[0] = np.asarray([8, 17])
    expected_y[1] = np.asarray([9, 18])

    expected_architecture = get_expected_architecture(dataset.shape[1])

    assert np.array_equal(actual_X, expected_X)
    assert np.array_equal(actual_y, expected_y)
    assert actual_architecture.get_config() == expected_architecture.get_config()

    fccfg.OBSERVATION_WINDOW = actual_observation_window


def test_evaluate_none_forecaster():
    valid_dataset = pd.DataFrame({'col': ['value']})
    with pytest.raises(Exception) as e_info:
        CNNForecaster().evaluate(valid_dataset)
    assert str(e_info.value) == "You must train the forecaster before calling evaluate method"


def test_forecast_none_forecaster():
    valid_dataset = pd.DataFrame({'col': ['value']})
    with pytest.raises(Exception) as e_info:
        CNNForecaster().forecast(valid_dataset)
    assert str(e_info.value) == "You must train the forecaster before calling forecast method"


def get_expected_architecture(n_features):
    architecture = Sequential(name='cnn-lstm')
    architecture(TimeDistributed(
        Conv1D(filters=64, kernel_size=1, activation='relu'),
        input_shape=(None, fccfg.OBSERVATION_WINDOW, n_features)))
    architecture(TimeDistributed(MaxPooling1D(pool_size=2)))
    architecture(TimeDistributed(Flatten()))
    architecture.add(LSTM(
        name='lstm_1',
        input_shape=(fccfg.OBSERVATION_WINDOW, n_features), return_sequences=True,
        units=100))
    architecture.add(LSTM(
        name='lstm_2',
        input_shape=(fccfg.OBSERVATION_WINDOW, n_features),
        units=100))
    architecture.add(Dense(
        name='dense',
        units=n_features,
        activation='sigmoid'))
    architecture.compile(loss=fccfg.ERROR_METRIC, optimizer='adam', metrics=['mse', 'mae'])
    return architecture
