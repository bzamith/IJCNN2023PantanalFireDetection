"""
Module which contains the Forecaster class
It contains the required methods to train a forecasting model and extract forecasted data
"""
from typing import Any, List, Tuple

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

import config.data_preparation_settings as dpcfg
import config.forecast_settings as fccfg
import config.general_settings as cfg

from src.enum.forecasting_algorithms_enum import ForecastingAlgorithmEnum


class Forecaster:
    """The Forecaster entity"""

    algorithm: ForecastingAlgorithmEnum
    forecaster: Any

    def build_architecture(self, dataset: pd.DataFrame) -> Tuple[np.array, np.array, Any]:
        """
        Build the base architecture for the forecaster
        :param dataset: The dataset for training the forecaster
        :return: a list containing the assets X and y from dataset, and the forecaster base architecture
        """
        if self.__class__ == Forecaster:
            raise Exception("Class Forecaster must not be called directly")

    def learn(self, dataset: pd.DataFrame) -> Any:
        """
        Trains the forecasting model
        :param dataset: The dataset for training the forecaster
        :return: The history from forecaster
        """
        if self.__class__ == Forecaster:
            raise Exception("Class Forecaster must not be called directly")

        X, y, self.forecaster = self.build_architecture(dataset)
        self.forecaster.compile(loss=fccfg.ERROR_METRIC, optimizer='adam', metrics=['mse', 'mae'])

        callback = EarlyStopping(monitor='val_loss', patience=fccfg.EARLY_STOPPING_PATIENCE)

        history = self.forecaster.fit(X, y,
                                      epochs=fccfg.NB_EPOCHS,
                                      callbacks=[callback],
                                      verbose=fccfg.VERBOSE,
                                      validation_split=1 - dpcfg.TRAIN_SIZE,
                                      batch_size=1,
                                      shuffle=False)

        return history

    def evaluate(self, dataset: pd.DataFrame) -> List[float]:
        """
        Gets score for model in a test set
        :param dataset: The dataset for testing the model
        :return: The score
        """
        if self.__class__ == Forecaster:
            raise Exception("Class Forecaster must not be called directly")
        if self.forecaster is None:
            raise Exception("You must train the forecaster before calling evaluate method")

        assets = self.get_assets(dataset)
        X = assets[0]
        y = assets[1]
        scores = self.forecaster.evaluate(X, y)
        if not isinstance(scores, list):
            scores = [scores]
        return scores

    def forecast(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the forecasted data after training the model
        :param dataset: The dataset for which the value will be forecasted
        :return: The forecasted values
        """
        if self.__class__ == Forecaster:
            raise Exception("Class Forecaster must not be called directly")

        if self.forecaster is None:
            raise Exception("You must train the forecaster before calling forecast method")

        n_rows = dataset.shape[0]
        dataset = np.array(dataset)
        dataset, n_features = self.__reshape_dataset(dataset, (int(n_rows / fccfg.OBSERVATION_WINDOW)))

        forecasted = pd.DataFrame(self.forecaster.predict(dataset))
        return forecasted

    def get_assets(self, dataset: pd.DataFrame) -> Tuple[np.array, np.array, int]:
        """
        Extracts the assets given the dataset, for the forecaster
        :param dataset: The dataset from which the assets will be extracted
        :return: The input (x) and target (y), as well as the number of features
        """
        X, y = self.__split_sequence(np.asarray(dataset))
        n_rows = X.shape[0]
        X, n_features = self.__reshape_dataset(X, n_rows)
        return X, y, n_features

    def plot_history(self, history: Any, file_name: str) -> None:
        """
        Plots the history and saves
        :param history: the history
        :param file_name: the file name, without path
        """
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(cfg.OUTPUT_FIGURES_DIR + file_name)
        plt.clf()

    def plot_model(self, file_name: str) -> None:
        """
        Plots the model and saves
        :param file_name: the file name, without path
        """
        plot_model(
            self.forecaster,
            to_file=cfg.OUTPUT_FIGURES_DIR + file_name,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=True
        )

    def __split_sequence(self, sequence: np.array) -> Tuple[np.array, np.array]:
        """
        Extracts the input and target data for forecasting models
        :param sequence: The sequence that will be used by the forecaster
        :return: The input (x) and target (y) sequences for training
        """
        n_col = len(sequence[0])
        x, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + fccfg.OBSERVATION_WINDOW
            if end_ix > len(sequence) - 1:
                break
            seq_x = sequence[i:end_ix, 0:n_col]
            x.append(seq_x)
            seq_y = sequence[end_ix, 0:n_col]
            y.append(seq_y)
        return np.array(x), np.array(y)

    def __reshape_dataset(self, dataset: np.array, first_dim: int) -> Tuple[np.array, int]:
        """
        Reshapes the dataset accordingly, considering the forecasting window size
        :param dataset: The dataset that will be reshaped
        :param first_dim: The first dimension for reshaping
        :return: The reshaped dataset and the number of features value
        """
        try:
            n_features = dataset[0].shape[1]
        except IndexError:
            n_features = dataset.shape[1]
        reshaped_dataset = dataset.reshape((first_dim, fccfg.OBSERVATION_WINDOW, n_features))
        return reshaped_dataset, n_features
