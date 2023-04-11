from unittest import mock

import pandas as pd
import pytest

import config.general_settings as cfg
from src.data.datafile import Datafile
from src.enum.data_sources_enum import DataSourceEnum
from src.enum.file_extensions_enum import FileExtensionEnum

FILE_PATH = cfg.DATA_DIR
DATA_SOURCE = DataSourceEnum.INMET
FILE_NAME = "file.csv"
FILE_EXTENSION = ".csv"
FULL_FILE_PATH = FILE_PATH + DATA_SOURCE.value + '/' + FILE_NAME


@mock.patch('src.data.file_reader.csv_file_reader.pd')
def test_constructor_valid(mock_pd):
    valid_output = pd.DataFrame({'col': ['value']})
    datafile = Datafile(DATA_SOURCE, FILE_NAME)

    mock_pd.read_csv.return_value = valid_output
    mock_pd.read_csv.assert_called_with(datafile.full_file_path)

    assert datafile.file_name == FILE_NAME
    assert datafile.file_extension == FileExtensionEnum(FILE_EXTENSION)
    assert datafile.full_file_path == FULL_FILE_PATH
    assert datafile.data_source == DATA_SOURCE
    assert datafile.data.equals(valid_output)


def test_read_file_null_data_source():
    with pytest.raises(ValueError) as e_info:
        Datafile(None, FILE_NAME)
    assert str(e_info.value) == "Parameter data_source must not be null"


def test_read_file_data_source_invalid_type():
    with pytest.raises(TypeError) as e_info:
        Datafile("string", FILE_NAME)
    assert str(
        e_info.value) == "Parameter data_source must be of type DataSourceEnum"


def test_read_file_null_file_name():
    with pytest.raises(ValueError) as e_info:
        Datafile(DATA_SOURCE, None)
    assert str(e_info.value) == "Parameter file_name must not be null"


def test_constructor_blank_file_extension():
    file_name = "file"
    with pytest.raises(ValueError) as e_info:
        Datafile(DATA_SOURCE, file_name)
    assert str(e_info.value) == "File extension not found: " + file_name


def test_constructor_invalid_file_extension():
    file_name = "file.xxx"
    with pytest.raises(ValueError) as e_info:
        Datafile(DATA_SOURCE, file_name)
    assert str(e_info.value) == "File extension not recognized: " + ".xxx"
