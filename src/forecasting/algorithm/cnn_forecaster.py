"""
Module which contains the CNNForecaster class
It contains the required methods to train a CNN+LSTM based forecasting model and extract forecasted data
CNN stands for Convolutional Neural Network
LSTM stands for Long Short Term Memory
"""
from typing import Any, Tuple

import numpy as np

import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM, MaxPooling1D, TimeDistributed

import config.forecast_settings as fccfg

from src.enum.forecasting_algorithms_enum import ForecastingAlgorithmEnum
from src.forecasting.algorithm.forecaster import Forecaster


class CNNForecaster(Forecaster):
    """
    The CNNForecaster entity
    It extends the abstract Forecaster class
    """

    algorithm = ForecastingAlgorithmEnum.CNN
    forecaster = None

    def build_architecture(self, dataset: pd.DataFrame) -> Tuple[np.array, np.array, Any]:
        """
        Build the base architecture for the forecaster
        :param dataset: The dataset for training the forecaster
        :return: a list containing the assets X and y from dataset, and the forecaster base architecture
        """
        super(CNNForecaster, self).build_architecture(dataset)
        architecture = Sequential(name='cnn-lstm')

        X, y, n_features = self.get_assets(dataset)

        architecture(TimeDistributed(
            Conv1D(filters=64, kernel_size=1, activation='relu'),
            input_shape=(None, fccfg.OBSERVATION_WINDOW, n_features)))
        architecture(TimeDistributed(
            MaxPooling1D(pool_size=2)))
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

        return X, y, architecture

    def learn(self, dataset: pd.DataFrame) -> Any:
        """
        Trains the CNN+LSTM forecasting model
        :param dataset: The dataset for training the forecaster
        :return: The history from forecaster
        """
        history = super(CNNForecaster, self).learn(dataset)

        self.plot_history(history, "cnn_forecaster_history.png")
        # self.plot_model("cnn_forecaster_model.png")

        return history
