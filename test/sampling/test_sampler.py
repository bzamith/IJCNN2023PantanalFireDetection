import pandas as pd
import pytest

from src.sampling.sampler import Sampler


def test_constructor_call_directly_fit_sample():
    valid_x = pd.DataFrame({'col': ['value']})
    valid_y = pd.DataFrame({'target': ['0']})
    with pytest.raises(Exception) as e_info:
        Sampler().fit_sample(valid_x, valid_y)
    assert str(e_info.value) == "Class Sampler must not be called directly"
