"""
Module that has all operations required to create the forecasted dataset
that is, forecasting the instances, for forecasting algorithms based in ml
model (train + test)
"""
from typing import List

import pandas as pd

import config.dataset_settings as dscfg
import config.forecast_settings as fccfg
import config.general_settings as cfg

import src.utils.pandas_utils as pdutils
from src.forecasting.algorithm.forecaster import Forecaster


def create_forecasted_datasets(forecaster: Forecaster,
                               X_train: pd.DataFrame,
                               X_test: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Calls different methods sequentially in order to create the forecasted datasets
    for N days in the future
    :param forecaster: the forecaster object
    :param X_train: the ml X_test dataset, it is required to get the instances that will be used for learning (by selecting the dates)
    :param X_test: the ml X_test dataset, it is required to get the instances that will be forecasted (by selecting the dates)
    :return: list of forecasted datasets (one for each day in the future)
    """
    columns_ignore = dscfg.COLUMNS_IGNORE_FOR_ML

    dataset = pdutils.join_dataframes_x_wise([X_train, X_test])
    dataset = pdutils.delete_columns(dataset, columns_ignore)

    predict_start_index = X_train.shape[0]

    forecasted_datasets = []
    last_forecasted_dataset = pd.DataFrame()

    # Generates X different forecasted datasets, with X being equal to
    # FORECAST_HORIZON setting
    for i in range(0, fccfg.FORECAST_HORIZON):
        forecasted_dataset = __generate_forecasted_dataset(dataset,
                                                           predict_start_index,
                                                           X_test,
                                                           forecaster,
                                                           last_forecasted_dataset,
                                                           i)
        forecasted_datasets.append(forecasted_dataset)
        last_forecasted_dataset = forecasted_dataset

    # Adds Date column and saves dataset
    for i, forecasted_dataset in enumerate(forecasted_datasets):
        forecasted_datasets[i] = __fix_and_save_forecasted_dataset(forecasted_dataset,
                                                                   X_test,
                                                                   i)
    return forecasted_datasets


def __generate_forecasted_dataset(dataset: pd.DataFrame,
                                  predict_start_index: int,
                                  X_test: pd.DataFrame,
                                  forecaster: Forecaster,
                                  last_forecasted_dataset: pd.DataFrame,
                                  days_in_future: int) -> pd.DataFrame:
    """
    Calls different methods sequentially in order to generate the forecasted dataset
    :param dataset: the full dataset (train + test)
    :param predict_start_index: the index from which the prediction should start
    :param X_test: the ml X_test dataset, it is required to get the instances that will be forecasted (by selecting the dates)
    :param forecaster: the forecaster object
    :param last_forecasted_dataset: the dataset with the last forecasting results
    :param days_in_future: number of days in the future that is to be forecasted
    :return: the forecasted dataset
    """
    forecasted_data = pd.DataFrame()
    for i in range(0, days_in_future):
        last_forecasted_row = pdutils.select_rows(last_forecasted_dataset, i)
        forecasted_data = pdutils.join_dataframes_x_wise([forecasted_data, last_forecasted_row])

    for i in range(predict_start_index + days_in_future, predict_start_index + X_test.shape[0]):
        x_actual = pdutils.select_rows(dataset, range(i - fccfg.OBSERVATION_WINDOW, i - days_in_future))
        if last_forecasted_dataset.empty:
            x = x_actual
        else:
            index_start = i - predict_start_index - 1
            selected_indexes = range(index_start, index_start + days_in_future)
            x_last_forecasted = pdutils.select_rows(last_forecasted_dataset, selected_indexes)
            x = pdutils.join_dataframes_x_wise([x_actual, x_last_forecasted], drop_column_names=True)

        forecasted = forecaster.forecast(x)
        forecasted_data = pd.concat([forecasted_data, forecasted])

    return forecasted_data


def __fix_and_save_forecasted_dataset(forecasted_dataset: pd.DataFrame,
                                      X_test: pd.DataFrame,
                                      curr_day_in_the_future: int) -> pd.DataFrame:
    """
    Fixes and then saves the forecasted dataset
    :param forecasted_dataset: the dataset with forecasted data
    :param X_test: the ml X_test dataset, it is required to get the instances that will be forecasted (by selecting the dates)
    :param curr_day_in_the_future: the day in the future day is currently being forecasted
    :return: the forecasted dataset
    """
    dates = pdutils.select_columns(X_test, dscfg.DATE_COLUMN_NAME, reset_row_indexes=True)

    forecasted_dataset = pdutils.join_dataframes_y_wise([dates, forecasted_dataset])

    if forecasted_dataset.shape != X_test.shape:
        raise Exception("forecasted_dataset shape doesn't match X_test shape")

    forecasted_dataset.columns = X_test.columns

    forecasted_dataset.to_csv(cfg.DATA_GENERATED_DIR + "forecasted_dataset_" + str(curr_day_in_the_future) + ".csv")

    return forecasted_dataset
