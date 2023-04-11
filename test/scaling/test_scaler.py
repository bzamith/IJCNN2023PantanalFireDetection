import pandas as pd
import pytest

from src.scaling.scaler import Scaler


def test_constructor_call_directly_fit_scale():
    valid_dataset = pd.DataFrame({'col': ['value']})
    valid_columns = ['col', 'col2']
    with pytest.raises(Exception) as e_info:
        Scaler().fit_scale(valid_dataset, valid_columns)
    assert str(
        e_info.value) == "Class Scaler must not be called directly"


def test_constructor_call_directly_scale():
    valid_dataset = pd.DataFrame({'col': ['value']})
    valid_columns = ['col', 'col2']
    with pytest.raises(Exception) as e_info:
        Scaler().scale(valid_dataset, valid_columns)
    assert str(
        e_info.value) == "Class Scaler must not be called directly"


def test_constructor_call_directly_descale():
    valid_dataset = pd.DataFrame({'col': ['value']})
    valid_columns = ['col', 'col2']
    with pytest.raises(Exception) as e_info:
        Scaler().descale(valid_dataset, valid_columns)
    assert str(
        e_info.value) == "Class Scaler must not be called directly"
