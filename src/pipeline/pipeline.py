"""Module which represents the pipeline and its operations"""
from typing import List, Union

import config.general_settings as cfg

from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum
from src.enum.forecasting_algorithms_enum import ForecastingAlgorithmEnum
from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.enum.scaling_methods_enum import ScalingMethodEnum
from src.pipeline.step import StepOutput
from src.utils import validation_utils


class FitPipelineParameters:
    """Input for pipeline generate models execution"""

    sampling_methods: List[SamplingMethodEnum]
    scaling_methods: List[ScalingMethodEnum]
    classification_algorithms: List[ClassificationAlgorithmEnum]
    forecasters: List[ForecastingAlgorithmEnum]

    def __init__(self,
                 scaling_methods: List[str],
                 forecasting_algorithms: List[str],
                 sampling_methods: List[str],
                 classification_algorithms: List[str]
                 ):
        """Class constructor"""
        self.scaling_methods = []
        for scaling_method in scaling_methods:
            validation_utils.validate_enum(scaling_method, "Scaling Method", ScalingMethodEnum)
            self.scaling_methods.append(ScalingMethodEnum(scaling_method))
        self.forecasting_algorithms = []
        for forecast_algorithm in forecasting_algorithms:
            validation_utils.validate_enum(forecast_algorithm, "Forecasting Algorithm", ForecastingAlgorithmEnum)
            self.forecasting_algorithms.append(ForecastingAlgorithmEnum(forecast_algorithm))
        self.sampling_methods = []
        for sampling_method in sampling_methods:
            validation_utils.validate_enum(sampling_method, "Sampling Method", SamplingMethodEnum)
            self.sampling_methods.append(SamplingMethodEnum(sampling_method))
        self.classification_algorithms = []
        for classification_algorithm in classification_algorithms:
            validation_utils.validate_enum(classification_algorithm, "Classification Algorithm", ClassificationAlgorithmEnum)
            self.classification_algorithms.append(ClassificationAlgorithmEnum(classification_algorithm))


class PredictPipelineParameters:
    """Input for pipeline predict execution"""

    def __init__(self):
        """Class constructor"""
        self.validate_input()

    def validate_input(self) -> None:
        """
        Validate the input for the pipeline
        """
        try:
            pass
        except Exception as exception:
            raise exception


class Pipeline:
    """The Pipeline entity"""

    def run(self, execution_parameters: Union[FitPipelineParameters, PredictPipelineParameters]) -> StepOutput:
        """
        Run pipeline, according to the sequential pipeline defined
        :param execution_parameters: The parameters for each run of the pipeline
        :return: The output of the last step
        """
        if self.__class__ == Pipeline:
            raise Exception("Class Pipeline must not be called directly")
        return None

    def save_elapsed_time(self, start: float, end: float) -> None:
        """
        Saves pipeline elapsed run time
        """
        with open(cfg.ASSETS_DIR + "elapsed_time.txt", 'w') as f:
            elapsed_time = end - start
            f.write("Pipeline elapsed run time: ")
            f.write("\n")
            f.write(str(elapsed_time))
