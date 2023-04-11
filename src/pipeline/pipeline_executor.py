"""Module which contains operations to execute pipeline"""
import config.classification_settings as clfcfg
import config.data_preparation_settings as dpcfg
import config.forecast_settings as fccfg

from src.enum.pipelines_enum import PipelineEnum
from src.pipeline import pipeline_factory
from src.pipeline.pipeline import FitPipelineParameters, PredictPipelineParameters


def execute_fit_pipeline() -> None:
    """Run pipeline fit"""
    pipeline_parameters = FitPipelineParameters(
        dpcfg.SCALING_METHOD,
        fccfg.ALGORITHM,
        dpcfg.SAMPLING_METHOD,
        clfcfg.ALGORITHM
    )

    pipeline = pipeline_factory.get(PipelineEnum.FIT)
    pipeline.run(pipeline_parameters)


def execute_predict_pipeline() -> None:
    """Run pipeline predict"""
    pipeline_parameters = PredictPipelineParameters()

    pipeline = pipeline_factory.get(PipelineEnum.PREDICT)
    pipeline.run(pipeline_parameters)
