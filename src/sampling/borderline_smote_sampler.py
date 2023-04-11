"""
Module which contains the BorderlineSMOTESampler class
It contains the required methods to implement a BorderlineSMOTE Sampler
"""
from imblearn.over_sampling import BorderlineSMOTE

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class BorderlineSMOTESampler(Sampler):
    """The BorderlineSMOTESampler entity"""

    method = SamplingMethodEnum.BORDERLINE_SMOTE
    base_sampler = BorderlineSMOTE(random_state=cfg.SEED)
    sampler = None
