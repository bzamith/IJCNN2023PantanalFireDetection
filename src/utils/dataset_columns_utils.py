"""Module with utilities for getting column names"""
from typing import List

import config.dataset_settings as dscfg
import config.forecast_settings as fccfg

from src.enum.risk_rate_index_enum import RiskRateIndexEnum


def get_column_name_prefix(name: str, prefix: str) -> str:
    """
    Get formatted column name by appending a prefix
    :param name: the column name
    :param prefix: the prefix to be appended
    :return: the resulting column name
    """
    return prefix + dscfg.COMMON_SEPARATOR + name


def get_column_name_suffix(name: str, suffix: str) -> str:
    """
    Get formatted column name by appending a suffix
    :param name: the column name
    :param suffix: the suffix to be appended
    :return: the resulting column name
    """
    return name + dscfg.COMMON_SEPARATOR + suffix


def get_column_names_prefix(name: str, prefix: str) -> List[str]:
    """
    Get formatted column names by appending a prefix
    :param name: the common column name
    :param prefix: the prefix to be appended
    :return: the resulting column names list
    """
    if name == dscfg.PRESENT_COLUMN_NAME:
        return [prefix + dscfg.COMMON_SEPARATOR + name]
    if name == dscfg.FORECASTED_COLUMN_NAME:
        columns = []
        for i in range(0, fccfg.FORECAST_HORIZON):
            columns.append(prefix + dscfg.COMMON_SEPARATOR + name +
                           dscfg.COMMON_SEPARATOR + str(i))
        return columns


def get_column_names_suffix(name: str, suffix: str) -> List[str]:
    """
    Get formatted column names by appending a suffix
    :param name: the common column name
    :param suffix: the suffix to be appended
    :return: the resulting column names list
    """
    if name == dscfg.PRESENT_COLUMN_NAME:
        return [name + dscfg.COMMON_SEPARATOR + suffix]
    if name == dscfg.FORECASTED_COLUMN_NAME:
        columns = []
        for i in range(0, fccfg.FORECAST_HORIZON):
            columns.append(name + dscfg.COMMON_SEPARATOR + str(i) +
                           dscfg.COMMON_SEPARATOR + suffix)
        return columns


def get_column_names_prefix_suffix(name: str, prefix: str, suffix: str) -> List[str]:
    """
    Get formatted column names by appending a prefix and a suffix
    :param name: the common column name
    :param prefix: the prefix to be appended
    :param suffix: the suffix to be appended
    :return: the resulting column names list
    """
    if name == dscfg.PRESENT_COLUMN_NAME:
        return [prefix + dscfg.COMMON_SEPARATOR + name + dscfg.COMMON_SEPARATOR + suffix]
    if name == dscfg.FORECASTED_COLUMN_NAME:
        columns = []
        for i in range(0, fccfg.FORECAST_HORIZON):
            columns.append(prefix + dscfg.COMMON_SEPARATOR + name + dscfg.COMMON_SEPARATOR +
                           str(i) + dscfg.COMMON_SEPARATOR + suffix)
        return columns


def get_column_names_risk_rate_suffix(name: str, suffix: str) -> List[str]:
    """
    Get formatted column names for risk rates by appending a suffix
    :param name: the common column name
    :param suffix: the suffix to be appended
    :return: the resulting column names list
    """
    output = []
    if name == dscfg.PRESENT_COLUMN_NAME:
        for risk_rate in RiskRateIndexEnum:
            output.append(name + dscfg.COMMON_SEPARATOR + suffix + dscfg.COMMON_SEPARATOR +
                          risk_rate.value.lower())
    if name == dscfg.FORECASTED_COLUMN_NAME:
        for i in range(0, fccfg.FORECAST_HORIZON):
            for risk_rate in RiskRateIndexEnum:
                output.append(name + dscfg.COMMON_SEPARATOR + str(i) + dscfg.COMMON_SEPARATOR +
                              suffix + dscfg.COMMON_SEPARATOR + risk_rate.value.lower())
    return output
