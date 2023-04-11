"""Module which represents a factory for Pipeline"""

from src.enum.pipelines_enum import PipelineEnum
from src.exception.not_implemented_exception import NotImplementedException
from src.pipeline.pipeline import Pipeline
from src.pipeline.pipelines.fit_pipeline import FitPipeline
from src.pipeline.pipelines.predict_pipeline import PredictPipeline


def get(pipeline: PipelineEnum) -> Pipeline:
    """
    Factory method for PipelineEnum
    :param pipeline: The enum for the pipeline
    :return: The pipeline object for that given enum
    """
    if not pipeline:
        raise ValueError("Parameter pipeline must not be null")
    if not isinstance(pipeline, PipelineEnum):
        raise TypeError("Parameter pipeline must be of type PipelineEnum")
    if pipeline == PipelineEnum.FIT:
        return __fit_pipeline()
    if pipeline == PipelineEnum.PREDICT:
        return __predict_pipeline()
    raise NotImplementedException("No Pipeline implemented for pipeline {}".format(pipeline.value))


def __fit_pipeline() -> FitPipeline:
    return FitPipeline()


def __predict_pipeline() -> PredictPipeline:
    return PredictPipeline()
