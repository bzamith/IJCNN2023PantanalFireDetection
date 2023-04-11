from unittest import mock

from imblearn.combine import SMOTETomek

import pandas as pd
import pytest

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.smotetomek_sampler import SMOTETomekSampler


def test_attributes():
    sampler = SMOTETomekSampler()
    assert sampler.method == SamplingMethodEnum.SMOTETOMEK
    assert isinstance(sampler.base_sampler, SMOTETomek)
    assert sampler.sampler is None


@mock.patch('imblearn.combine.SMOTETomek.fit_resample')
def test_fit_sample(mock_sampling_fit):
    valid_x = pd.DataFrame({'col': ['value0', 'value1'],
                            'col1': ['value2', 'value3']})
    valid_y = pd.DataFrame({'col': ['0', '1']})

    return_x = pd.DataFrame({'col': ['value1', 'value0']})
    return_y = pd.DataFrame({'col': ['1', '0']})

    columns = ['col']

    mock_sampling_fit.return_value = return_x, return_y

    output_x, output_y = SMOTETomekSampler().fit_sample(valid_x, valid_y, columns=columns)

    assert mock_sampling_fit.call_args.args[0].equals(valid_x[columns])
    assert mock_sampling_fit.call_args.args[1].equals(valid_y)

    assert output_x.equals(return_x)
    assert output_y.equals(return_y)


def test_fit_sample_default_columns():
    valid_x = pd.DataFrame({'col': ['value0', 'value1'],
                            'col1': ['value2', 'value3']})
    valid_y = pd.DataFrame({'col': ['0', '1']})

    with pytest.raises(KeyError) as e_info:
        SMOTETomekSampler().fit_sample(valid_x, valid_y)
    assert str(e_info.value).__contains__("None of")
    assert str(e_info.value).__contains__("are in the [columns]")

