"""Module which represents a factory for Evaluator"""

from src.enum.evaluation_measures_enum import EvaluationMeasureEnum
from src.evaluation.measure.correlation_evaluator import CorrelationEvaluator
from src.evaluation.measure.evaluator import Evaluator
from src.exception.not_implemented_exception import NotImplementedException


def get(evaluation_measure: EvaluationMeasureEnum) -> Evaluator:
    """
    Factory method for EvaluationMeasures
    :param evaluation_measure: The enum for evaluation measure
    :return: The evaluator object for that given enum
    """
    if not evaluation_measure:
        raise ValueError("Parameter evaluation_measure must not be null")
    if not isinstance(evaluation_measure, EvaluationMeasureEnum):
        raise TypeError("Parameter evaluation_measure must be of type EvaluationMeasureEnum")
    if evaluation_measure == EvaluationMeasureEnum.CORRELATION:
        return CorrelationEvaluator()
    raise NotImplementedException("No EvaluationMeasureEnum implemented for evaluation measure {}".format(evaluation_measure.value))
