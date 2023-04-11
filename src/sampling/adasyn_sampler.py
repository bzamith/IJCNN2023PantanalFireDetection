"""
Module which contains the ADASYNSampler class
It contains the required methods to implement a ADASYN Sampler
"""
from imblearn.over_sampling import ADASYN

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class ADASYNSampler(Sampler):
    """The ADASYNSampler entity"""

    method = SamplingMethodEnum.ADASYN
    base_sampler = ADASYN(random_state=cfg.SEED)
    sampler = None
