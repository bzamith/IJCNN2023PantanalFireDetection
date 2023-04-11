"""Module which contains the RiskRateIndexEnum enum class"""

from src.enum.enum_class import EnumClass


class RiskRateIndexEnum(EnumClass):
    """Enum for risk rate classes"""

    def __new__(self, *args, **kwds):
        """Definition of RiskRateIndexEnum element initial args"""
        obj = object.__new__(self)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, prob_threshold: float, factor_value: int):
        """Constructor for RiskRateIndexEnum element attributes"""
        self._prob_threshold_ = prob_threshold
        self._factor_value_ = factor_value

    @property
    def prob_threshold(self) -> float:
        """
        Getter for prob_threshold
        :return: The threshold of the probability (0 to 1) so which it belongs to the index
        """
        return self._prob_threshold_

    @property
    def factor_value(self) -> int:
        """
        Getter for factor_value
        :return: The factor value of the index
        """
        return self._factor_value_

    def set_prob_threshold(self, threshold: float) -> None:
        """
        Setter for prob_threshold
        :param threshold: The new threshold that wants to be set to the index
        """
        self._prob_threshold_ = threshold

    # (Name, Default Threshold, Factor)
    NULO = ("Nulo", 0.05, 0)
    PEQUENO = ("Pequeno", 0.25, 1)
    MEDIO = ("Médio", 0.50, 2)
    ALTO = ("Alto", 0.75, 3)
    MUITO_ALTO = ("Muito alto", 1.00, 4)
