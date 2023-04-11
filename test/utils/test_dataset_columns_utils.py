import config.dataset_settings as dscfg
import config.forecast_settings as fccfg

import src.utils.dataset_columns_utils as dcutils

from src.enum.risk_rate_index_enum import RiskRateIndexEnum

NAME = "name"
PREFIX = "prefix"
SUFFIX = "suffix"
PRESENT = "present"
FORECASTED = "forecasted"
COMMON_SEPARATOR = dscfg.COMMON_SEPARATOR


def test_get_column_name_prefix():
    expected_output = PREFIX + COMMON_SEPARATOR + NAME
    assert dcutils.get_column_name_prefix(NAME, PREFIX) == expected_output


def test_get_column_name_suffix():
    expected_output = NAME + COMMON_SEPARATOR + SUFFIX
    assert dcutils.get_column_name_suffix(NAME, SUFFIX) == expected_output


def test_get_column_names_prefix_present():
    expected_output = [PREFIX + COMMON_SEPARATOR + PRESENT]
    assert dcutils.get_column_names_prefix(PRESENT, PREFIX) == expected_output


def test_get_column_names_prefix_forecasted():
    expected_output = []
    for i in range(0, fccfg.FORECAST_HORIZON):
        expected_output.append(PREFIX + COMMON_SEPARATOR + FORECASTED + COMMON_SEPARATOR + str(i))
    assert dcutils.get_column_names_prefix(FORECASTED, PREFIX) == expected_output


def test_get_column_names_suffix_present():
    expected_output = [PRESENT + COMMON_SEPARATOR + SUFFIX]
    assert dcutils.get_column_names_suffix(PRESENT, SUFFIX) == expected_output


def test_get_column_names_suffix_forecasted():
    expected_output = []
    for i in range(0, fccfg.FORECAST_HORIZON):
        expected_output.append(FORECASTED + COMMON_SEPARATOR + str(i) + COMMON_SEPARATOR + SUFFIX)
    assert dcutils.get_column_names_suffix(FORECASTED, SUFFIX) == expected_output


def test_get_column_names_prefix_suffix_present():
    expected_output = [PREFIX + COMMON_SEPARATOR + PRESENT + COMMON_SEPARATOR + SUFFIX]
    assert dcutils.get_column_names_prefix_suffix(PRESENT, PREFIX, SUFFIX) == expected_output


def test_get_column_names_prefix_suffix_forecasted():
    expected_output = []
    for i in range(0, fccfg.FORECAST_HORIZON):
        expected_output.append(PREFIX + COMMON_SEPARATOR + FORECASTED + COMMON_SEPARATOR + str(i) + COMMON_SEPARATOR + SUFFIX)
    assert dcutils.get_column_names_prefix_suffix(FORECASTED, PREFIX, SUFFIX) == expected_output


def test_get_column_names_risk_rate_suffix_present():
    expected_output = []
    for risk_rate in RiskRateIndexEnum:
        expected_output.append(PRESENT + COMMON_SEPARATOR + SUFFIX + COMMON_SEPARATOR + risk_rate.value.lower())
    assert dcutils.get_column_names_risk_rate_suffix(PRESENT, SUFFIX) == expected_output


def test_get_column_names_risk_rate_suffix_forecasted():
    expected_output = []
    for i in range(0, fccfg.FORECAST_HORIZON):
        for risk_rate in RiskRateIndexEnum:
            expected_output.append(FORECASTED + COMMON_SEPARATOR + str(i) + COMMON_SEPARATOR + SUFFIX + COMMON_SEPARATOR + risk_rate.value.lower())
    assert dcutils.get_column_names_risk_rate_suffix(FORECASTED, SUFFIX) == expected_output

