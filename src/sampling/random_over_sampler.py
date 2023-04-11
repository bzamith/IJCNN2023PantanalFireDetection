"""
Module which contains the RandomOverSampler class
It contains the required methods to implement a Random Over Sampler
"""
import imblearn.over_sampling

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class RandomOverSampler(Sampler):
    """The RandomOverSampler entity"""

    method = SamplingMethodEnum.RANDOM_OVER_SAMPLER
    base_sampler = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority', random_state=cfg.SEED)
    sampler = None
