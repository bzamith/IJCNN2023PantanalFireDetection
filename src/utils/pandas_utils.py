"""Module with utilities for handling pandas dataframes"""
from typing import Any, Hashable, List, Union

import pandas as pd

import config.dataset_settings as dscfg


def select_columns(dataframe: pd.DataFrame,
                   columns: Union[str, int, List[Union[str, int]]],
                   reset_row_indexes: bool = False) -> pd.DataFrame:
    """
    Select columns given a list of column names, from a given data.
    It does not change the dataframe that was passed as input. Instead, it generates
    a new one and returns
    :param dataframe: the input dataframe
    :param columns: the column names or indexes to be selected
    :param reset_row_indexes: whether to reset row indexes or not
    :return: the new dataframe
    """
    if isinstance(columns, str) or isinstance(columns, int):
        columns = [columns]
    for i in range(0, len(columns)):
        if isinstance(columns[i], int):
            columns[i] = dataframe.columns[columns[i]]
    if reset_row_indexes:
        return dataframe[columns].reset_index(drop=True)
    return dataframe[columns]


def select_rows(dataframe: pd.DataFrame,
                rows: Union[int, List[int], range]) -> pd.DataFrame:
    """
    Select rows given a list of indexes, from a given data.
    It does not change the dataframe that was passed as input. Instead, it generates
    a new one and returns. Row indexing starts with 0
    :param dataframe: the input dataframe
    :param rows: the row indexes to be selected
    :return: the new dataframe
    """
    if isinstance(rows, int):
        rows = [rows]
    return dataframe.iloc[rows].reset_index(drop=True)


def select_rows_by_value(dataframe: pd.DataFrame,
                         column: Union[str, int],
                         value: Any) -> pd.DataFrame:
    """
    Select rows for which a given column that has specified value.

    It does not change the dataframe that was passed as input. Instead, it generates
    a new one and returns
    :param dataframe: the input dataframe
    :param column: can be either the column name or the column index
    :param value: the value to use as reference for selecting the row
    :return: the new dataframe
    """
    if isinstance(column, str):
        column = column
    else:
        column = dataframe.columns[column]
    return dataframe.loc[dataframe[column] == value].reset_index(drop=True)


def delete_columns(dataframe: pd.DataFrame,
                   columns: Union[str, int, List[Union[str, int]]]) -> pd.DataFrame:
    """
    Delete a given column from dataframe.
    It does not change the dataframe that was passed as input. Instead, it generates
    a new one and returns
    :param dataframe: the input dataframe
    :param columns: the column names or indexes to be deleted
    :return: the new dataframe
    """
    if isinstance(columns, str) or isinstance(columns, int):
        columns = [columns]
    for i in range(0, len(columns)):
        if isinstance(columns[i], int):
            columns[i] = dataframe.columns[columns[i]]
    return dataframe.drop(columns, axis=1).reset_index(drop=True)


def select_value_row_column(dataframe: pd.DataFrame,
                            row: Union[int, Hashable],
                            column: Union[str, int]) -> Any:
    """
    Select a value from dataframe given the row and the column.
    :param dataframe: the input dataframe
    :param row: the row index
    :param column: the column name or index
    :return: the selected value
    """
    if isinstance(column, int):
        column = dataframe.columns[column]
    return dataframe.loc[dataframe.index[row], column]


def join_dataframes_x_wise(dataframes: List[pd.DataFrame],
                           drop_column_names: bool = False) -> pd.DataFrame:
    """
    Join two dataframes row-wise (x-axis).
    :param dataframes: the list of dataframes to be joined x-wise
    :param drop_column_names: whether to drop the column names of not
    :return: the new dataframe
    """
    for dataframe in dataframes:
        dataframe.reset_index(drop=True, inplace=True)

    if drop_column_names:
        for dataframe in dataframes:
            dataframe.columns = [''] * len(dataframe.columns)

    output_dataframe = None
    for dataframe in dataframes:
        if output_dataframe is None:
            output_dataframe = dataframe
        else:
            output_dataframe = pd.concat([output_dataframe, dataframe.reset_index(drop=True)], ignore_index=True)
    return output_dataframe


def join_dataframes_y_wise(dataframes: List[pd.DataFrame],
                           drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Join two dataframes column-wise (y-axis).
    :param dataframes: the list of dataframes to be joined y-wise
    :param drop_duplicates: whether to drop duplicated columns or not
    :return: the new dataframe
    """
    column_names = []
    for dataframe in dataframes:
        dataframe.reset_index(drop=True, inplace=True)
        column_names += list(dataframe.columns)

    output_dataframe = None
    for dataframe in dataframes:
        if output_dataframe is None:
            output_dataframe = dataframe
        else:
            output_dataframe = pd.concat([output_dataframe, dataframe.reset_index(drop=True)], ignore_index=True, axis=1)

    output_dataframe.columns = column_names
    if drop_duplicates:
        output_dataframe = output_dataframe.loc[:, ~output_dataframe.columns.duplicated()]
    return output_dataframe


def remove_duplicated_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated columns in a dataframe.
    It does not change the dataframe that was passed as input. Instead, it generates
    a new one and returns
    :param dataframe: the input dataframe
    :return: the new dataframe
    """
    return dataframe.loc[:, ~dataframe.columns.duplicated()]


def select_columns_with_substring(dataframe: pd.DataFrame,
                                  substrings: Union[str, List[str]]) -> pd.DataFrame:
    """
    Select columns in a dataframe whose name has given substrings.
    It does not change the dataframe that was passed as input. Instead, it generates
    a new one and returns
    :param dataframe: the input dataframe
    :param substrings: the list of substrings
    :return: the new dataframe
    """
    if isinstance(substrings, str):
        substrings = [substrings]
    output_dataframe = dataframe.filter(regex=substrings[0])
    for i in range(1, len(substrings)):
        output_dataframe = join_dataframes_y_wise([output_dataframe, dataframe.filter(regex=substrings[i])])
    return output_dataframe


def add_row(dataframe: pd.DataFrame,
            values: List[Any]) -> pd.DataFrame:
    """
    Add a new row to dataframe given a list of values.
    It does not change the dataframe that was passed as input. Instead, it generates
    a new one and returns
    :param dataframe: the input dataframe
    :param values: the list of values for the new row
    :return: the new dataframe
    """
    dataframe.loc[len(dataframe)] = values
    return dataframe


def dataframe_to_list(dataframe: pd.DataFrame) -> List[Any]:
    """
    Create a list from a dataframe with single column.
    :param dataframe: the input dataframe
    :return: the list of values from the dataframe
    """
    if dataframe.shape[1] > 1:
        raise ValueError("Parameter dataframe must have one single column")
    column_name = dataframe.columns[0]
    return dataframe[column_name].to_list()


def add_prefix_to_column_names(dataframe: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Add a prefix to all column names.
    It does not change the dataframe that was passed as input. Instead, it generates
    a new one and returns
    :param dataframe: the input dataframe
    :param prefix: the prefix to be added
    :return: the new dataframe
    """
    new_column_names = []
    output_dataframe = dataframe.copy()
    for column_name in dataframe.columns:
        new_column_names.append(prefix + column_name)
    output_dataframe.columns = new_column_names
    return output_dataframe


def join_inner_by_date(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Inner join dataframes by date column
    :param dataframes: the list of input dataframes
    :return: the new dataframe
    """
    date_column_name = dscfg.DATE_COLUMN_NAME

    for i in range(0, len(dataframes)):
        if date_column_name not in dataframes[i].columns:
            raise ValueError("Parameter dataframe " + str(i) + " does not contain date column")

    output_dataframe = dataframes[0].copy()

    for i in range(1, len(dataframes)):
        output_dataframe = pd.merge(output_dataframe.reset_index(drop=True),
                                    dataframes[i].reset_index(drop=True),
                                    on=date_column_name,
                                    how='inner')
    return output_dataframe
