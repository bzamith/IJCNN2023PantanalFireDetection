"""
Module which contains the DeployModelStep, DeployModelStepInput and DeployModelStepOutput classes
They contain the required methods to run the prediction algorithm (ml)
"""
from pickle import dump
from typing import List

import config.general_settings as cfg

from src.classification.algorithm.classifier import Classifier
from src.forecasting.algorithm.forecaster import Forecaster
from src.pipeline.step import Step, StepInput, StepOutput
from src.sampling.sampler import Sampler, SamplingMethodEnum
from src.scaling.scaler import Scaler


class DeployModelStep(Step):
    """The DeployModelStep entity"""

    step_name = "Deploy Model"
    step_description = "Deploy model assets"

    def __init__(self,
                 scaler: Scaler,
                 forecaster: Forecaster,
                 sampler: Sampler,
                 classifier: Classifier,
                 thresholds: List[float]):
        """
        Class constructor
        :param scaler: the fitted scaler
        :param forecaster: the trained forecaster
        :param sampler: the fitted sampler
        :param classifier: the trained classifier
        :param thresholds: the selected thresholds
        """
        self.step_input = DeployModelStepInput(scaler, forecaster, sampler, classifier, thresholds)
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""

        definitions_dict = {
            cfg.SCALING_METHOD_KEY: self.step_input.scaler.method,
            cfg.FORECASTING_ALGORITHM_KEY: self.step_input.forecaster.algorithm,
            cfg.SAMPLING_METHOD_KEY: self.step_input.sampler.method,
            cfg.CLASSIFICATION_ALGORITHM_KEY: self.step_input.classifier.algorithm,
            cfg.THRESHOLDS_KEY: self.step_input.thresholds,
        }

        save_file_format = "{}{}"
        path = cfg.OUTPUT_DEPLOYED_MODEL_DIR

        dump(definitions_dict, open(save_file_format.format(path, cfg.DEFINITIONS_DICT_FILE_NAME), 'wb'))
        dump(self.step_input.scaler.scaler, open(save_file_format.format(path, cfg.SCALER_FILE_NAME), 'wb'))
        self.step_input.forecaster.forecaster.save(save_file_format.format(path, cfg.FORECASTER_SUB_DIR))
        if self.step_input.sampler.method != SamplingMethodEnum.NONE:
            dump(self.step_input.sampler.sampler, open(save_file_format.format(path, cfg.SAMPLER_FILE_NAME), 'wb'))
        dump(self.step_input.classifier.classifier, open(save_file_format.format(path, cfg.CLASSIFIER_FILE_NAME), 'wb'))

        self.step_output = DeployModelStepOutput()


class DeployModelStepInput(StepInput):
    """Input for DeployModelStep"""

    scaler: Scaler
    forecaster: Forecaster
    sampler: Sampler
    classifier: Classifier
    thresholds: List[float]

    def __init__(self,
                 scaler: Scaler,
                 forecaster: Forecaster,
                 sampler: Sampler,
                 classifier: Classifier,
                 thresholds: List[float]):
        """
        Class constructor
        :param scaler: the fitted scaler
        :param forecaster: the trained forecaster
        :param sampler: the fitted sampler
        :param classifier: the trained classifier
        :param thresholds: the selected thresholds
        """
        self.scaler = scaler
        self.forecaster = forecaster
        self.sampler = sampler
        self.classifier = classifier
        self.thresholds = thresholds


class DeployModelStepOutput(StepOutput):
    """Output for DeployModelStep"""
