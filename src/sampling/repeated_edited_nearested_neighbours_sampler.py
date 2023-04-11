"""
Module which contains the RepeatedEditedNearestNeighboursSampler class
It contains the required methods to implement a RepeatedEditedNearestNeighbours Sampler
"""
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class RepeatedEditedNearestNeighboursSampler(Sampler):
    """The RepeatedEditedNearestNeighboursSampler entity"""

    method = SamplingMethodEnum.REPEATED_EDITED_NEAREST_NEIGHBOURS
    base_sampler = RepeatedEditedNearestNeighbours()
    sampler = None
