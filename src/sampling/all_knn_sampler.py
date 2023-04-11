"""
Module which contains the AllKNNSampler class
It contains the required methods to implement a AllKNN Sampler
"""
from imblearn.under_sampling import AllKNN

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class AllKNNSampler(Sampler):
    """The AllKNNSampler entity"""

    method = SamplingMethodEnum.ALL_KNN
    base_sampler = AllKNN()
    sampler = None
