"""Module which contains the PipelineEnum enum class"""

from src.enum.enum_class import EnumClass


class PipelineEnum(EnumClass):
    """Enum for different pipelines"""

    FIT = "Fit Models"
    PREDICT = "Predict Values"
