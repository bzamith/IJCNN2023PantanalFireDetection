"""
Module which contains the GRUForecaster class
It contains the required methods to train a GRU based forecasting model and extract forecasted data
GRU stands for Gated Recurrent Unit
"""
from typing import Any, Tuple

import numpy as np

import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU

import config.forecast_settings as fccfg

from src.enum.forecasting_algorithms_enum import ForecastingAlgorithmEnum
from src.forecasting.algorithm.forecaster import Forecaster


class GRUForecaster(Forecaster):
    """
    The GRUForecaster entity
    It extends the abstract Forecaster class
    """

    algorithm = ForecastingAlgorithmEnum.GRU
    forecaster = None

    def build_architecture(self, dataset: pd.DataFrame) -> Tuple[np.array, np.array, Any]:
        """
        Build the base architecture for the forecaster
        :param dataset: The dataset for training the forecaster
        :return: a list containing the assets X and y from dataset, and the forecaster base architecture
        """
        super(GRUForecaster, self).build_architecture(dataset)
        architecture = Sequential(name='gru')

        X, y, n_features = self.get_assets(dataset)

        architecture.add(GRU(
            name='gru_1',
            input_shape=(fccfg.OBSERVATION_WINDOW, n_features), return_sequences=True,
            units=100))
        architecture.add(GRU(
            name='gru_2',
            input_shape=(fccfg.OBSERVATION_WINDOW, n_features),
            units=100))
        architecture.add(Dense(
            name='dense',
            units=n_features,
            activation='sigmoid'))

        return X, y, architecture

    def learn(self, dataset: pd.DataFrame) -> None:
        """
        Trains the GRU forecasting model
        :param dataset: The dataset for training the forecaster
        :return: The history from forecaster
        """
        history = super(GRUForecaster, self).learn(dataset)

        self.plot_history(history, "gru_forecaster_history.png")
        # self.plot_model("gru_forecaster_model.png")

        return history
