import pytest

from src.data.file_reader.file_reader import FileReader


def test_constructor_call_directly():
    valid_datafile = "valid123.csv"
    with pytest.raises(Exception) as e_info:
        FileReader().read_data(valid_datafile)
    assert str(e_info.value) == "Class FileReader must not be called directly"
