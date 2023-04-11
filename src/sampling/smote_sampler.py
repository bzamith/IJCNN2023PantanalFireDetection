"""
Module which contains the SMOTESampler class
It contains the required methods to implement a SMOTE Sampler
"""
from imblearn.over_sampling import SMOTE

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class SMOTESampler(Sampler):
    """The SMOTESampler entity"""

    method = SamplingMethodEnum.SMOTE
    base_sampler = SMOTE(random_state=cfg.SEED)
    sampler = None
