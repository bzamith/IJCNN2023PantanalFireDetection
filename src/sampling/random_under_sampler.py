"""
Module which contains the RandomUnderSampler class
It contains the required methods to implement a Random Under Sampler
"""
import imblearn.under_sampling

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class RandomUnderSampler(Sampler):
    """The RandomUnderSampler entity"""

    method = SamplingMethodEnum.RANDOM_UNDER_SAMPLER
    base_sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority', random_state=cfg.SEED)
    sampler = None
