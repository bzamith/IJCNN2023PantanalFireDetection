"""
Module which contains the Sampler class
It contains the required methods to handle the imbalanced data by performing sampling
"""
import copy
from typing import Any, List, Tuple

import pandas as pd

import config.dataset_settings as dscfg

from src.enum.sampling_methods_enum import SamplingMethodEnum


class Sampler:
    """The Sampler entity"""

    method: SamplingMethodEnum
    base_sampler: Any
    sampler: Any

    def fit_sample(self, X: pd.DataFrame,
                   y: pd.DataFrame,
                   columns: List[str] = dscfg.COLUMNS_SAMPLING) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits and samples the data
        :param X: the dataframe containing the attributes
        :param y: the dataframe containing the targets
        :param columns: the columns to be kept
        :return: X sampled and y sampled
        """
        if self.__class__ == Sampler:
            raise Exception("Class Sampler must not be called directly")

        self.sampler = copy.deepcopy(self.base_sampler)
        X_sampled, y_sampled = self.sampler.fit_resample(X[columns], y)
        return X_sampled, y_sampled
