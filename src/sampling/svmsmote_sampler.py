"""
Module which contains the SVMSMOTESampler class
It contains the required methods to implement a SVMSMOTE Sampler
"""
from imblearn.over_sampling import SVMSMOTE

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class SVMSMOTESampler(Sampler):
    """The SVMSMOTESampler entity"""

    method = SamplingMethodEnum.SVMSMOTE
    base_sampler = SVMSMOTE(random_state=cfg.SEED)
    sampler = None
