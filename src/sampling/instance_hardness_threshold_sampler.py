"""
Module which contains the InstanceHardnessThresholdSampler class
It contains the required methods to implement a InstanceHardnessThreshold Sampler
"""
from imblearn.under_sampling import InstanceHardnessThreshold

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class InstanceHardnessThresholdSampler(Sampler):
    """The InstanceHardnessThresholdSampler entity"""

    method = SamplingMethodEnum.INSTANCE_HARDNESS_THRESHOLD
    base_sampler = InstanceHardnessThreshold(random_state=cfg.SEED)
    sampler = None
