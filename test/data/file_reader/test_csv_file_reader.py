from unittest import mock

import pandas as pd
import pytest

from src.data.file_reader.csv_file_reader import CSVFileReader

FULL_PATH = "/path/to/file/file.csv"


@mock.patch('src.data.file_reader.csv_file_reader.pd')
def test_read_data_valid_csv(mock_pd):
    valid_output = pd.DataFrame({'col': ['value']})

    mock_pd.read_csv.return_value = valid_output

    assert valid_output.equals(CSVFileReader().read_data(FULL_PATH))

    mock_pd.read_csv.assert_called_with(FULL_PATH)


def test_constructor_not_data_csv():
    xlsx_full_file_path = "valid123.xlsx"

    with pytest.raises(ValueError) as e_info:
        CSVFileReader().read_data(xlsx_full_file_path)
    assert str(
        e_info.value) == "File extension .xlsx doesn't match reader extension .csv"


def test_constructor_null_file():
    null_datafile = None
    with pytest.raises(ValueError) as e_info:
        CSVFileReader().read_data(null_datafile)
    assert str(e_info.value) == "Parameter full_file_path must not be null"


def test_constructor_blank_extension():
    no_extension_file = "valid123."
    with pytest.raises(ValueError) as e_info:
        CSVFileReader().read_data(no_extension_file)
    assert str(
        e_info.value) == "File extension not identified for file from path: " + no_extension_file
