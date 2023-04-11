from unittest.mock import MagicMock

import pytest

from src.enum.risk_rate_algorithms_enum import RiskRateAlgorithmEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.risk_rate.algorithm import risk_rate_algorithm_factory
from src.risk_rate.algorithm.angstron_risk_rate import AngstronRiskRate
from src.risk_rate.algorithm.fma_plus_risk_rate import FMAPlusRiskRate
from src.risk_rate.algorithm.fma_risk_rate import FMARiskRate
from src.risk_rate.algorithm.nesterov_risk_rate import NesterovRiskRate
from src.risk_rate.algorithm.telicyn_risk_rate import TelicynRiskRate


def test_get_fma():
    algorithm = risk_rate_algorithm_factory.get(
        RiskRateAlgorithmEnum.FMA)
    assert isinstance(algorithm, FMARiskRate)


def test_get_nesterov():
    algorithm = risk_rate_algorithm_factory.get(RiskRateAlgorithmEnum.NESTEROV)
    assert isinstance(algorithm, NesterovRiskRate)


def test_get_telicyn():
    algorithm = risk_rate_algorithm_factory.get(RiskRateAlgorithmEnum.TELICYN)
    assert isinstance(algorithm, TelicynRiskRate)


def test_get_fma_plus():
    algorithm = risk_rate_algorithm_factory.get(RiskRateAlgorithmEnum.FMA_PLUS)
    assert isinstance(algorithm, FMAPlusRiskRate)


def test_get_angstron():
    algorithm = risk_rate_algorithm_factory.get(RiskRateAlgorithmEnum.ANGSTRON)
    assert isinstance(algorithm, AngstronRiskRate)


def test_get_invalid():
    with pytest.raises(TypeError) as e_info:
        risk_rate_algorithm_factory.get("xxx")
    assert str(e_info.value) == "Parameter risk_rate_algorithm must be of type RiskRateAlgorithmEnum"


def test_get_none():
    with pytest.raises(ValueError) as e_info:
        risk_rate_algorithm_factory.get(None)
    assert str(e_info.value) == "Parameter risk_rate_algorithm must not be null"


def test_get_not_implemented():
    mock = MagicMock(spec=RiskRateAlgorithmEnum, name="Dummy", value="DummyValue")
    with pytest.raises(NotImplementedException) as e_info:
        risk_rate_algorithm_factory.get(mock)
    assert str(e_info.value) == "No RiskRateAlgorithmEnum implemented for risk rate algorithm DummyValue"
