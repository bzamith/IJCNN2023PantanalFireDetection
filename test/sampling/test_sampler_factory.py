from unittest.mock import MagicMock

import pytest

from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.sampling import sampler_factory
from src.sampling.adasyn_sampler import ADASYNSampler
from src.sampling.all_knn_sampler import AllKNNSampler
from src.sampling.borderline_smote_sampler import BorderlineSMOTESampler
from src.sampling.cluster_centroids_sampler import ClusterCentroidsSampler
from src.sampling.edited_nearest_neighbours_sampler import EditedNearestNeighboursSampler
from src.sampling.instance_hardness_threshold_sampler import InstanceHardnessThresholdSampler
from src.sampling.near_miss_sampler import NearMissSampler
from src.sampling.neighbourhood_cleaning_rule_sampler import NeighbourhoodCleaningRuleSampler
from src.sampling.none_sampler import NoneSampler
from src.sampling.one_sided_selection_sampler import OneSidedSelectionSampler
from src.sampling.random_over_sampler import RandomOverSampler
from src.sampling.random_under_sampler import RandomUnderSampler
from src.sampling.repeated_edited_nearested_neighbours_sampler import RepeatedEditedNearestNeighboursSampler
from src.sampling.smote_sampler import SMOTESampler
from src.sampling.smoteenn_sampler import SMOTEENNSampler
from src.sampling.smotetomek_sampler import SMOTETomekSampler
from src.sampling.svmsmote_sampler import SVMSMOTESampler


def test_get_none():
    algorithm = sampler_factory.get(SamplingMethodEnum.NONE)
    assert isinstance(algorithm, NoneSampler)


def test_get_random_over_sampler():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.RANDOM_OVER_SAMPLER)
    assert isinstance(algorithm, RandomOverSampler)


def test_get_smote():
    algorithm = sampler_factory.get(SamplingMethodEnum.SMOTE)
    assert isinstance(algorithm, SMOTESampler)


def test_get_adasyn():
    algorithm = sampler_factory.get(SamplingMethodEnum.ADASYN)
    assert isinstance(algorithm, ADASYNSampler)


def test_get_borderline_smote():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.BORDERLINE_SMOTE)
    assert isinstance(algorithm, BorderlineSMOTESampler)


def test_get_svmsmote():
    algorithm = sampler_factory.get(SamplingMethodEnum.SVMSMOTE)
    assert isinstance(algorithm, SVMSMOTESampler)


def test_get_random_under_sampler():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.RANDOM_UNDER_SAMPLER)
    assert isinstance(algorithm, RandomUnderSampler)


def test_get_cluster_centroids():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.CLUSTER_CENTROIDS)
    assert isinstance(algorithm, ClusterCentroidsSampler)


def test_get_near_miss():
    algorithm = sampler_factory.get(SamplingMethodEnum.NEAR_MISS)
    assert isinstance(algorithm, NearMissSampler)


def test_get_edited_nearest_neighbours():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.EDITED_NEAREST_NEIGHBOURS)
    assert isinstance(algorithm, EditedNearestNeighboursSampler)


def test_get_repeated_edited_nearest_neighbours():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.REPEATED_EDITED_NEAREST_NEIGHBOURS)
    assert isinstance(algorithm, RepeatedEditedNearestNeighboursSampler)


def test_get_all_knn():
    algorithm = sampler_factory.get(SamplingMethodEnum.ALL_KNN)
    assert isinstance(algorithm, AllKNNSampler)


def test_get_one_sided_selection():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.ONE_SIDED_SELECTION)
    assert isinstance(algorithm, OneSidedSelectionSampler)


def test_get_neighbourhood_cleaning_rule():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.NEIGHBOURHOOD_CLEANING_RULE)
    assert isinstance(algorithm, NeighbourhoodCleaningRuleSampler)


def test_get_instance_hardness_threshold():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.INSTANCE_HARDNESS_THRESHOLD)
    assert isinstance(algorithm, InstanceHardnessThresholdSampler)


def test_get_smoteenn():
    algorithm = sampler_factory.get(SamplingMethodEnum.SMOTEENN)
    assert isinstance(algorithm, SMOTEENNSampler)


def test_get_smotetomek():
    algorithm = sampler_factory.get(
        SamplingMethodEnum.SMOTETOMEK)
    assert isinstance(algorithm, SMOTETomekSampler)


def test_get_invalid():
    with pytest.raises(TypeError) as e_info:
        sampler_factory.get("xxx")
    assert str(
        e_info.value) == "Parameter sampling_method must be of type SamplingMethodEnum"


def test_get_null():
    with pytest.raises(ValueError) as e_info:
        sampler_factory.get(None)
    assert str(e_info.value) == "Parameter sampling_method must not be null"


def test_get_not_implemented():
    mock = MagicMock(spec=SamplingMethodEnum, name="Dummy", value="DummyValue")
    with pytest.raises(NotImplementedException) as e_info:
        sampler_factory.get(mock)
    assert str(e_info.value) == "No SamplingMethodEnum implemented for sampling algorithm DummyValue"
