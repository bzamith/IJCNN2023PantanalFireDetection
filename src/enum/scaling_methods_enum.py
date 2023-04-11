"""Module which contains the ScalingMethodEnum enum class"""

from src.enum.enum_class import EnumClass


class ScalingMethodEnum(EnumClass):
    """Enum for different scaling (normalization) algorithms"""

    NONE = "None"
    MIN_MAX_SCALER = "Min Max Scaler"
    STANDARD_SCALER = "Standard Scaler"
    ROBUST_SCALER = "Robust Scaler"
    MAX_ABS_SCALER = "Max Abs Scaler"
