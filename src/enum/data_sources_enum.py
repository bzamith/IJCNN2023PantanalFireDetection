"""Module which contains the DataSourceEnum enum class"""

from src.enum.enum_class import EnumClass


class DataSourceEnum(EnumClass):
    """Enum for different data sources"""

    INMET = "inmet"
    HOTSPOT_DATA = "hotspot_data"
    CLIMATIC_DATA = "climatic_data"
    PREDICTION_DATA = "prediction_data"
