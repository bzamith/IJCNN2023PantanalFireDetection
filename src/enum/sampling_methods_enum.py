"""Module which contains the SamplingMethodEnum enum class"""

from src.enum.enum_class import EnumClass


class SamplingMethodEnum(EnumClass):
    """
    Enum for different sampling algorithms
    Reference: https://imbalanced-learn.org/stable/introduction.html
    """

    NONE = "None"
    # Over-sampling
    RANDOM_OVER_SAMPLER = "Random Over Sampler"
    SMOTE = "SMOTE"
    ADASYN = "ADASYN"
    BORDERLINE_SMOTE = "Borderline SMOTE"
    SVMSMOTE = "SVMSMOTE"
    # Under-sampling
    RANDOM_UNDER_SAMPLER = "Random Under Sampler"
    CLUSTER_CENTROIDS = "Cluster Centroids"
    NEAR_MISS = "Near Miss"
    EDITED_NEAREST_NEIGHBOURS = "Edited Nearest Neighbours"
    REPEATED_EDITED_NEAREST_NEIGHBOURS = "Repeated Edited Nearest Neighbours"
    ALL_KNN = "All KNN"
    ONE_SIDED_SELECTION = "One Sided Selection"
    NEIGHBOURHOOD_CLEANING_RULE = "Neighbourhood Cleaning Rule"
    INSTANCE_HARDNESS_THRESHOLD = "Instance Hardness Threshold"
    # Combination of over- and under-sampling
    SMOTEENN = "SMOTEENN"
    SMOTETOMEK = "SMOTETomek"
