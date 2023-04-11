"""
Module which contains the ClusterCentroidsSampler class
It contains the required methods to implement a ClusterCentroids Sampler
"""
from imblearn.under_sampling import ClusterCentroids

import config.general_settings as cfg

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.sampling.sampler import Sampler


class ClusterCentroidsSampler(Sampler):
    """The ClusterCentroidsSampler entity"""

    method = SamplingMethodEnum.CLUSTER_CENTROIDS
    base_sampler = ClusterCentroids(random_state=cfg.SEED)
    sampler = None
