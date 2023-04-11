from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.classification.algorithm.classifier import Classifier


def test_constructor_call_directly_train():
    valid_x = pd.DataFrame({'col': ['value']})
    valid_y = pd.DataFrame({'target': ['0']})
    with pytest.raises(Exception) as e_info:
        Classifier().train(valid_x, valid_y)
    assert str(e_info.value) == "Class Classifier must not be called directly"


def test_constructor_call_directly_predict():
    valid_x = pd.DataFrame({'col': ['value']})
    with pytest.raises(Exception) as e_info:
        Classifier().predict(valid_x)
    assert str(e_info.value) == "Class Classifier must not be called directly"


def test_constructor_call_directly_evaluate():
    valid_x = pd.DataFrame({'col': ['value']})
    valid_y = pd.DataFrame({'target': ['0']})
    with pytest.raises(Exception) as e_info:
        Classifier().evaluate(valid_x, valid_y)
    assert str(e_info.value) == "Class Classifier must not be called directly"
