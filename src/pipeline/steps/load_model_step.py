"""
Module which contains the LoadModelStep, LoadModelStepInput and LoadModelStepOutput classes
They contain the required methods to run the prediction algorithm (ml)
"""
from pickle import load
from typing import List

from tensorflow.keras.models import load_model

import config.general_settings as cfg

from src.classification.algorithm import classifier_factory
from src.classification.algorithm.classifier import Classifier
from src.forecasting.algorithm import forecaster_factory
from src.forecasting.algorithm.forecaster import Forecaster
from src.pipeline.step import Step, StepInput, StepOutput
from src.sampling import sampler_factory
from src.sampling.sampler import Sampler, SamplingMethodEnum
from src.scaling import scaler_factory
from src.scaling.scaler import Scaler


class LoadModelStep(Step):
    """The LoadModelStep entity"""

    step_name = "Load Model"
    step_description = "Load model assets"

    def __init__(self):
        """
        Class constructor
        """
        self.step_input = LoadModelStepInput()
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""

        load_file_format = "{}{}"
        path = cfg.OUTPUT_DEPLOYED_MODEL_DIR

        definitions_dict = load(open(load_file_format.format(path, cfg.DEFINITIONS_DICT_FILE_NAME), 'rb'))

        scaler = scaler_factory.get(definitions_dict[cfg.SCALING_METHOD_KEY])
        scaler.scaler = load(open(load_file_format.format(path, cfg.SCALER_FILE_NAME), 'rb'))

        forecaster = forecaster_factory.get(definitions_dict[cfg.FORECASTING_ALGORITHM_KEY])
        forecaster.forecaster = load_model(load_file_format.format(path, cfg.FORECASTER_SUB_DIR))

        sampler = sampler_factory.get(definitions_dict[cfg.SAMPLING_METHOD_KEY])
        if sampler.method != SamplingMethodEnum.NONE:
            sampler.sampler = load(open(load_file_format.format(path, cfg.SAMPLER_FILE_NAME), 'rb'))

        classifier = classifier_factory.get(definitions_dict[cfg.CLASSIFICATION_ALGORITHM_KEY])
        classifier.classifier = load(open(load_file_format.format(path, cfg.CLASSIFIER_FILE_NAME), 'rb'))

        thresholds = definitions_dict[cfg.THRESHOLDS_KEY]

        self.step_output = LoadModelStepOutput(scaler, forecaster, sampler, classifier, thresholds)


class LoadModelStepInput(StepInput):
    """Input for LoadModelStep"""


class LoadModelStepOutput(StepOutput):
    """Output for LoadModelStep"""

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
