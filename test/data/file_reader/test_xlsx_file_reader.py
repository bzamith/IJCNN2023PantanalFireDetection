from unittest import mock

import pandas as pd
import pytest

from src.data.file_reader.xlsx_file_reader import XLSXFileReader

FULL_PATH = "/path/to/file/file.xlsx"


@mock.patch('src.data.file_reader.xlsx_file_reader.pd')
def test_read_data_valid_csv(mock_pd):
    valid_output = pd.DataFrame({'col': ['value']})
    all_sheets = {'key': valid_output,
                  'key2': None}

    mock_pd.read_excel.return_value = all_sheets

    assert valid_output.equals(XLSXFileReader().read_data(FULL_PATH))

    mock_pd.read_excel.assert_called_with(FULL_PATH, sheet_name=None)


@mock.patch('src.data.file_reader.xlsx_file_reader.pd')
def test_read_data_valid_xlsx_sheet_name(mock_pd):
    valid_output = pd.DataFrame({'col': ['value']})
    sheet_name = "sheet_name"

    mock_pd.read_excel.return_value = valid_output

    assert valid_output.equals(
        XLSXFileReader().read_data_with_sheet_name(FULL_PATH, sheet_name))

    mock_pd.read_excel.assert_called_with(FULL_PATH, sheet_name=sheet_name)


def test_constructor_not_data_xslx():
    txt_file = "valid123.txt"
    with pytest.raises(ValueError) as e_info:
        XLSXFileReader().read_data(txt_file)
    assert str(
        e_info.value) == "File extension .txt doesn't match reader extension .xlsx"


def test_constructor_null_file():
    with pytest.raises(ValueError) as e_info:
        XLSXFileReader().read_data(None)
    assert str(e_info.value) == "Parameter full_file_path must not be null"


def test_constructor_blank_extension():
    no_extension_file = "valid123."
    with pytest.raises(ValueError) as e_info:
        XLSXFileReader().read_data(no_extension_file)
    assert str(
        e_info.value) == "File extension not identified for file from path: " + no_extension_file
