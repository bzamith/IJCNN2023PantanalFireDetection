from unittest.mock import MagicMock

import pytest

from src.enum.evaluation_measures_enum import EvaluationMeasureEnum
from src.evaluation.measure import evaluator_factory
from src.evaluation.measure.correlation_evaluator import CorrelationEvaluator
from src.exception.not_implemented_exception import NotImplementedException


def test_get_correlation():
    measure = evaluator_factory.get(EvaluationMeasureEnum.CORRELATION)
    assert isinstance(measure, CorrelationEvaluator)


def test_get_invalid():
    with pytest.raises(TypeError) as e_info:
        evaluator_factory.get("xxx")
    assert str(
        e_info.value) == "Parameter evaluation_measure must be of type EvaluationMeasureEnum"


def test_get_none():
    with pytest.raises(ValueError) as e_info:
        evaluator_factory.get(None)
    assert str(e_info.value) == "Parameter evaluation_measure must not be null"


def test_get_not_implemented():
    mock = MagicMock(spec=EvaluationMeasureEnum, name="Dummy", value="DummyValue")
    with pytest.raises(NotImplementedException) as e_info:
        evaluator_factory.get(mock)
    assert str(e_info.value) == "No EvaluationMeasureEnum implemented for evaluation measure DummyValue"
