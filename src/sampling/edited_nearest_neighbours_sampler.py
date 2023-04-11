"""
Module which contains the EditedNearestNeighboursSampler class
It contains the required methods to implement a EditedNearestNeighbours Sampler
"""
from imblearn.under_sampling import EditedNearestNeighbours

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class EditedNearestNeighboursSampler(Sampler):
    """The EditedNearestNeighboursSampler entity"""

    method = SamplingMethodEnum.EDITED_NEAREST_NEIGHBOURS
    base_sampler = EditedNearestNeighbours()
    sampler = None
