"""
Module which contains the SMOTEENNSampler class
It contains the required methods to implement a SMOTEENN Sampler
"""
from imblearn.combine import SMOTEENN

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class SMOTEENNSampler(Sampler):
    """The SMOTEENNSampler entity"""

    method = SamplingMethodEnum.SMOTEENN
    base_sampler = SMOTEENN(random_state=cfg.SEED)
    sampler = None
