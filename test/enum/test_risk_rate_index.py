from src.enum.risk_rate_index_enum import RiskRateIndexEnum


def test_risk_rate_nulo():
    risk_rate = RiskRateIndexEnum.NULO

    assert risk_rate.value == "Nulo"
    assert risk_rate.prob_threshold == 0.05
    assert risk_rate.factor_value == 0


def test_risk_rate_pequeno():
    risk_rate = RiskRateIndexEnum.PEQUENO

    assert risk_rate.value == "Pequeno"
    assert risk_rate.prob_threshold == 0.25
    assert risk_rate.factor_value == 1


def test_risk_rate_medio():
    risk_rate = RiskRateIndexEnum.MEDIO

    assert risk_rate.value == "Médio"
    assert risk_rate.prob_threshold == 0.5
    assert risk_rate.factor_value == 2


def test_risk_rate_alto():
    risk_rate = RiskRateIndexEnum.ALTO

    assert risk_rate.value == "Alto"
    assert risk_rate.prob_threshold == 0.75
    assert risk_rate.factor_value == 3


def test_risk_rate_muito_alto():
    risk_rate = RiskRateIndexEnum.MUITO_ALTO

    assert risk_rate.value == "Muito alto"
    assert risk_rate.prob_threshold == 1
    assert risk_rate.factor_value == 4


def test_set_threshold_risk_rate_nulo():
    risk_rate = RiskRateIndexEnum.NULO
    new_threshold = 0.00005
    risk_rate.set_prob_threshold(new_threshold)

    assert risk_rate.value == "Nulo"
    assert risk_rate.prob_threshold == new_threshold
    assert risk_rate.factor_value == 0


def test_set_threshold_risk_rate_pequeno():
    risk_rate = RiskRateIndexEnum.PEQUENO
    new_threshold = 0.00005
    risk_rate.set_prob_threshold(new_threshold)

    assert risk_rate.value == "Pequeno"
    assert risk_rate.prob_threshold == new_threshold
    assert risk_rate.factor_value == 1


def test_set_threshold_risk_rate_medio():
    risk_rate = RiskRateIndexEnum.MEDIO
    new_threshold = 0.00005
    risk_rate.set_prob_threshold(new_threshold)

    assert risk_rate.value == "Médio"
    assert risk_rate.prob_threshold == new_threshold
    assert risk_rate.factor_value == 2


def test_set_threshold_risk_rate_alto():
    risk_rate = RiskRateIndexEnum.ALTO
    new_threshold = 0.00005
    risk_rate.set_prob_threshold(new_threshold)

    assert risk_rate.value == "Alto"
    assert risk_rate.prob_threshold == new_threshold
    assert risk_rate.factor_value == 3


def test_set_threshold_risk_rate_muito_alto():
    risk_rate = RiskRateIndexEnum.MUITO_ALTO
    new_threshold = 0.00005
    risk_rate.set_prob_threshold(new_threshold)

    assert risk_rate.value == "Muito alto"
    assert risk_rate.prob_threshold == new_threshold
    assert risk_rate.factor_value == 4
