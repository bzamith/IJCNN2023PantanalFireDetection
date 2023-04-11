"""Module which contains the NoneSampler class"""
from typing import List, Tuple

import pandas as pd

import config.dataset_settings as dscfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class NoneSampler(Sampler):
    """The NoneSampler entity"""

    method = SamplingMethodEnum.NONE
    base_sampler = None
    sampler = None

    def fit_sample(self, X: pd.DataFrame,
                   y: pd.DataFrame,
                   columns: List[str] = dscfg.COLUMNS_SAMPLING) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        No sampling
        :param X: the dataframe containing the attributes
        :param y: the dataframe containing the targets
        :param columns: the columns to be kept
        :return: X sampled and y sampled
        """
        return X[columns], y
