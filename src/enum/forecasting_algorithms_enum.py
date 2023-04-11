"""Module which contains the ForecastingAlgorithmEnum enum class"""

from src.enum.enum_class import EnumClass


class ForecastingAlgorithmEnum(EnumClass):
    """Enum for different forecasting algorithms"""

    LSTM = "Long Short Term Memory"
    GRU = "Gated Recurrent Unit"
    CNN = "Convolutional Neural Network"
