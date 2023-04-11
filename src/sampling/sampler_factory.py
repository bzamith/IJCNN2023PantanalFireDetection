"""Module which represents a factory for Sampler"""

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.sampling.adasyn_sampler import ADASYNSampler
from src.sampling.all_knn_sampler import AllKNNSampler
from src.sampling.borderline_smote_sampler import BorderlineSMOTESampler
from src.sampling.cluster_centroids_sampler import ClusterCentroidsSampler
from src.sampling.edited_nearest_neighbours_sampler import \
    EditedNearestNeighboursSampler
from src.sampling.instance_hardness_threshold_sampler import \
    InstanceHardnessThresholdSampler
from src.sampling.near_miss_sampler import NearMissSampler
from src.sampling.neighbourhood_cleaning_rule_sampler import \
    NeighbourhoodCleaningRuleSampler
from src.sampling.none_sampler import NoneSampler
from src.sampling.one_sided_selection_sampler import OneSidedSelectionSampler
from src.sampling.random_over_sampler import RandomOverSampler
from src.sampling.random_under_sampler import RandomUnderSampler
from src.sampling.repeated_edited_nearested_neighbours_sampler import \
    RepeatedEditedNearestNeighboursSampler
from src.sampling.sampler import Sampler
from src.sampling.smote_sampler import SMOTESampler
from src.sampling.smoteenn_sampler import SMOTEENNSampler
from src.sampling.smotetomek_sampler import SMOTETomekSampler
from src.sampling.svmsmote_sampler import SVMSMOTESampler


def get(sampling_method: SamplingMethodEnum) -> Sampler:
    """
    Factory method for SamplingMethods
    :param sampling_method: the sampling method
    :return: the sampler for that sampling method
    """
    if not sampling_method:
        raise ValueError("Parameter sampling_method must not be null")
    if not isinstance(sampling_method, SamplingMethodEnum):
        raise TypeError("Parameter sampling_method must be of type SamplingMethodEnum")
    if sampling_method == SamplingMethodEnum.NONE:
        return __none()
    if sampling_method == SamplingMethodEnum.RANDOM_OVER_SAMPLER:
        return __roversampler()
    if sampling_method == SamplingMethodEnum.SMOTE:
        return __smote()
    if sampling_method == SamplingMethodEnum.ADASYN:
        return __adasyn()
    if sampling_method == SamplingMethodEnum.BORDERLINE_SMOTE:
        return __borderline_smote()
    if sampling_method == SamplingMethodEnum.SVMSMOTE:
        return __svmsmote()
    if sampling_method == SamplingMethodEnum.RANDOM_UNDER_SAMPLER:
        return __rundersampler()
    if sampling_method == SamplingMethodEnum.CLUSTER_CENTROIDS:
        return __cluster_centroids()
    if sampling_method == SamplingMethodEnum.NEAR_MISS:
        return __near_miss()
    if sampling_method == SamplingMethodEnum.EDITED_NEAREST_NEIGHBOURS:
        return __editednearestneighbours()
    if sampling_method == SamplingMethodEnum. \
            REPEATED_EDITED_NEAREST_NEIGHBOURS:
        return __repeditednearestneighbours()
    if sampling_method == SamplingMethodEnum.ALL_KNN:
        return __allknn()
    if sampling_method == SamplingMethodEnum.ONE_SIDED_SELECTION:
        return __onesidedselection()
    if sampling_method == SamplingMethodEnum.NEIGHBOURHOOD_CLEANING_RULE:
        return __neighbourhoodcleaningrule()
    if sampling_method == SamplingMethodEnum.INSTANCE_HARDNESS_THRESHOLD:
        return __instance_hardness_threshold()
    if sampling_method == SamplingMethodEnum.SMOTEENN:
        return __smoteenn()
    if sampling_method == SamplingMethodEnum.SMOTETOMEK:
        return __smotetomek()
    raise NotImplementedException("No SamplingMethodEnum implemented for sampling algorithm {}"
                                  .format(sampling_method.value))


def __none() -> NoneSampler:
    return NoneSampler()


def __roversampler() -> RandomOverSampler:
    return RandomOverSampler()


def __smote() -> SMOTESampler:
    return SMOTESampler()


def __adasyn() -> ADASYNSampler:
    return ADASYNSampler()


def __borderline_smote() -> BorderlineSMOTESampler:
    return BorderlineSMOTESampler()


def __svmsmote() -> SVMSMOTESampler:
    return SVMSMOTESampler()


def __rundersampler() -> RandomUnderSampler:
    return RandomUnderSampler()


def __cluster_centroids() -> ClusterCentroidsSampler:
    return ClusterCentroidsSampler()


def __near_miss() -> NearMissSampler:
    return NearMissSampler()


def __editednearestneighbours() -> EditedNearestNeighboursSampler:
    return EditedNearestNeighboursSampler()


def __repeditednearestneighbours() -> RepeatedEditedNearestNeighboursSampler:
    return RepeatedEditedNearestNeighboursSampler()


def __allknn() -> AllKNNSampler:
    return AllKNNSampler()


def __onesidedselection() -> OneSidedSelectionSampler:
    return OneSidedSelectionSampler()


def __neighbourhoodcleaningrule() -> NeighbourhoodCleaningRuleSampler:
    return NeighbourhoodCleaningRuleSampler()


def __instance_hardness_threshold() -> InstanceHardnessThresholdSampler:
    return InstanceHardnessThresholdSampler()


def __smoteenn() -> SMOTEENNSampler:
    return SMOTEENNSampler()


def __smotetomek() -> SMOTETomekSampler:
    return SMOTETomekSampler()
