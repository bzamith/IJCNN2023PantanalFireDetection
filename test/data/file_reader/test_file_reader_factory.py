from unittest.mock import MagicMock

import pytest

from src.data.file_reader import file_reader_factory
from src.data.file_reader.csv_file_reader import CSVFileReader
from src.data.file_reader.xlsx_file_reader import XLSXFileReader
from src.enum.file_extensions_enum import FileExtensionEnum
from src.exception.not_implemented_exception import NotImplementedException


def test_get_csv():
    file_reader = file_reader_factory.get(FileExtensionEnum.CSV)
    assert isinstance(file_reader, CSVFileReader)


def test_get_xlsx():
    file_reader = file_reader_factory.get(FileExtensionEnum.XLSX)
    assert isinstance(file_reader, XLSXFileReader)


def test_get_invalid():
    with pytest.raises(TypeError) as e_info:
        file_reader_factory.get("xxx")
    assert str(
        e_info.value) == "Parameter file_extension must be of type FileExtensionEnum"


def test_get_none():
    with pytest.raises(ValueError) as e_info:
        file_reader_factory.get(None)
    assert str(e_info.value) == "Parameter file_extension must not be null"


def test_get_not_implemented():
    mock = MagicMock(spec=FileExtensionEnum, name="Dummy", value="DummyValue")
    with pytest.raises(NotImplementedException) as e_info:
        file_reader_factory.get(mock)
    assert str(e_info.value) == "No FileReader implemented for extension DummyValue"
