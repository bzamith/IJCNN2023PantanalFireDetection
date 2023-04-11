"""
Module which contains the SMOTETomekSampler class
It contains the required methods to implement a SMOTETomek Sampler
"""
from imblearn.combine import SMOTETomek

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class SMOTETomekSampler(Sampler):
    """The SMOTETomekSampler entity"""

    method = SamplingMethodEnum.SMOTETOMEK
    base_sampler = SMOTETomek(random_state=cfg.SEED)
    sampler = None
