"""
Module which contains the OneSidedSelectionSampler class
It contains the required methods to implement a OneSidedSelection Sampler
"""
from imblearn.under_sampling import OneSidedSelection

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class OneSidedSelectionSampler(Sampler):
    """The OneSidedSelectionSampler entity"""

    method = SamplingMethodEnum.ONE_SIDED_SELECTION
    base_sampler = OneSidedSelection(random_state=cfg.SEED)
    sampler = None
