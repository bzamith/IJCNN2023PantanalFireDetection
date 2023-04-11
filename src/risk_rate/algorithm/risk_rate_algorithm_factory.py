"""Module which represents a factory for RiskRateAlgorithm"""

from src.enum.risk_rate_algorithms_enum import RiskRateAlgorithmEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.risk_rate.algorithm.angstron_risk_rate import AngstronRiskRate
from src.risk_rate.algorithm.fma_plus_risk_rate import FMAPlusRiskRate
from src.risk_rate.algorithm.fma_risk_rate import FMARiskRate
from src.risk_rate.algorithm.nesterov_risk_rate import NesterovRiskRate
from src.risk_rate.algorithm.risk_rate_algorithm import RiskRateAlgorithm
from src.risk_rate.algorithm.telicyn_risk_rate import TelicynRiskRate


def get(risk_rate_algorithm: RiskRateAlgorithmEnum) -> RiskRateAlgorithm:
    """
    Factory method for RiskRateAlgorithms
    :param risk_rate_algorithm: the risk rate algorithm
    :return: the object corresponding to that risk rate algorithm
    """
    if not risk_rate_algorithm:
        raise ValueError("Parameter risk_rate_algorithm must not be null")
    if not isinstance(risk_rate_algorithm, RiskRateAlgorithmEnum):
        raise TypeError("Parameter risk_rate_algorithm must be of type RiskRateAlgorithmEnum")
    if risk_rate_algorithm == RiskRateAlgorithmEnum.FMA:
        return FMARiskRate()
    if risk_rate_algorithm == RiskRateAlgorithmEnum.NESTEROV:
        return NesterovRiskRate()
    if risk_rate_algorithm == RiskRateAlgorithmEnum.TELICYN:
        return TelicynRiskRate()
    if risk_rate_algorithm == RiskRateAlgorithmEnum.FMA_PLUS:
        return FMAPlusRiskRate()
    if risk_rate_algorithm == RiskRateAlgorithmEnum.ANGSTRON:
        return AngstronRiskRate()
    raise NotImplementedException("No RiskRateAlgorithmEnum implemented for risk rate algorithm {}".format(risk_rate_algorithm.value))
