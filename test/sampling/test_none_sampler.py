import pandas as pd

import pytest

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.none_sampler import NoneSampler


def test_attributes():
    sampler = NoneSampler()
    assert sampler.method == SamplingMethodEnum.NONE
    assert sampler.base_sampler is None
    assert sampler.sampler is None


def test_fit_sample():
    valid_X = pd.DataFrame({'col': ['value0', 'value1'],
                            'col1': ['value2', 'value3']})
    valid_y = pd.DataFrame({'col': ['0', '1']})
    expected_X = pd.DataFrame({'col': ['value0', 'value1']})

    columns = ['col']

    output_X, output_y = NoneSampler().fit_sample(valid_X, valid_y, columns=columns)

    assert output_X.equals(expected_X)
    assert output_y.equals(valid_y)


def test_fit_sample_default_columns():
    valid_X = pd.DataFrame({'col': ['value0', 'value1'],
                            'col1': ['value2', 'value3']})
    valid_y = pd.DataFrame({'col': ['0', '1']})

    with pytest.raises(KeyError) as e_info:
        NoneSampler().fit_sample(valid_X, valid_y)
    assert str(e_info.value).__contains__("None of")
    assert str(e_info.value).__contains__("are in the [columns]")

