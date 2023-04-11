"""
Module which contains the NearMissSampler class
It contains the required methods to implement a NearMiss Sampler
"""
from imblearn.under_sampling import NearMiss

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class NearMissSampler(Sampler):
    """The NearMissSampler entity"""

    method = SamplingMethodEnum.NEAR_MISS
    base_sampler = NearMiss()
    sampler = None
