"""
Module which contains the NeighbourhoodCleaningRuleSampler class
It contains the required methods to implement a NeighbourhoodCleaningRule Sampler
"""
from imblearn.under_sampling import NeighbourhoodCleaningRule

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class NeighbourhoodCleaningRuleSampler(Sampler):
    """The NeighbourhoodCleaningRuleSampler entity"""

    method = SamplingMethodEnum.NEIGHBOURHOOD_CLEANING_RULE
    base_sampler = NeighbourhoodCleaningRule()
    sampler = None
