from unittest.mock import MagicMock

import pytest

from src.enum.forecasting_algorithms_enum import ForecastingAlgorithmEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.forecasting.algorithm import forecaster_factory
from src.forecasting.algorithm.cnn_forecaster import CNNForecaster
from src.forecasting.algorithm.gru_forecaster import GRUForecaster
from src.forecasting.algorithm.lstm_forecaster import LSTMForecaster


def test_get_lstm():
    algorithm = forecaster_factory.get(ForecastingAlgorithmEnum.LSTM)
    assert isinstance(algorithm, LSTMForecaster)


def test_get_gru():
    algorithm = forecaster_factory.get(ForecastingAlgorithmEnum.GRU)
    assert isinstance(algorithm, GRUForecaster)


def test_get_cnn():
    algorithm = forecaster_factory.get(ForecastingAlgorithmEnum.CNN)
    assert isinstance(algorithm, CNNForecaster)


def test_get_invalid():
    with pytest.raises(TypeError) as e_info:
        forecaster_factory.get("xxx")
    assert str(e_info.value) == "Parameter forecast_algorithm must be of type ForecastingAlgorithmEnum"


def test_get_none():
    with pytest.raises(ValueError) as e_info:
        forecaster_factory.get(None)
    assert str(e_info.value) == "Parameter forecast_algorithm must not be null"


def test_get_not_implemented():
    mock = MagicMock(spec=ForecastingAlgorithmEnum, name="Dummy", value="DummyValue")
    with pytest.raises(NotImplementedException) as e_info:
        forecaster_factory.get(mock)
    assert str(e_info.value) == "No Forecaster implemented for forecasting algorithm DummyValue"