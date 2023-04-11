from src.enum.risk_rate_index_enum import RiskRateIndexEnum


def test_is_member():
    assert RiskRateIndexEnum.is_member(RiskRateIndexEnum.NULO.value)


def test_is_not_member():
    assert not RiskRateIndexEnum.is_member("dummy")
