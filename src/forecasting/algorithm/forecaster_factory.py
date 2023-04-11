"""Module which represents a factory for Forecaster"""

from src.enum.forecasting_algorithms_enum import ForecastingAlgorithmEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.forecasting.algorithm.cnn_forecaster import CNNForecaster
from src.forecasting.algorithm.forecaster import Forecaster
from src.forecasting.algorithm.gru_forecaster import GRUForecaster
from src.forecasting.algorithm.lstm_forecaster import LSTMForecaster


def get(forecast_algorithm: ForecastingAlgorithmEnum) -> Forecaster:
    """
    Factory method for ForecastAlgorithms
    :param forecast_algorithm: The enum for forecasting algorithm
    :return: The forecaster object for that given enum
    """
    if not forecast_algorithm:
        raise ValueError("Parameter forecast_algorithm must not be null")
    if not isinstance(forecast_algorithm, ForecastingAlgorithmEnum):
        raise TypeError("Parameter forecast_algorithm must be of type ForecastingAlgorithmEnum")
    if forecast_algorithm == ForecastingAlgorithmEnum.LSTM:
        return LSTMForecaster()
    if forecast_algorithm == ForecastingAlgorithmEnum.GRU:
        return GRUForecaster()
    if forecast_algorithm == ForecastingAlgorithmEnum.CNN:
        return CNNForecaster()
    raise NotImplementedException("No Forecaster implemented for forecasting algorithm {}".format(forecast_algorithm.value))
