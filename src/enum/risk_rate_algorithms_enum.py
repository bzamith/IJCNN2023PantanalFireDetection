"""Module which contains the RiskRateAlgorithmEnum enum class"""

from src.enum.enum_class import EnumClass


class RiskRateAlgorithmEnum(EnumClass):
    """Enum for different risk rate algorithms"""

    FMA = "fma"
    FMA_PLUS = "fma_plus"
    TELICYN = "telicyn"
    ANGSTRON = "angstron"
    NESTEROV = "nesterov"
