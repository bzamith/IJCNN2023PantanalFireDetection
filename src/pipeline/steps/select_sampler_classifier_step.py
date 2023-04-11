"""
Module which contains the SelectSamplerClassifierStep, SelectSamplerClassifierStepInput and SelectSamplerClassifierStepOutput classes
They contain the required methods to sample the datasets and train the classifier
"""
from typing import List, Tuple

import pandas as pd

import config.dataset_settings as dscfg
import config.general_settings as cfg

import src.utils.pandas_utils as pdutils
from src.classification.algorithm import classifier_factory
from src.classification.algorithm.classifier import Classifier
from src.enum.classification_algorithms_enum import ClassificationAlgorithmEnum
from src.enum.sampling_methods_enum import SamplingMethodEnum
from src.pipeline.step import Step, StepInput, StepOutput
from src.sampling import sampler_factory
from src.sampling.sampler import Sampler


class SelectSamplerClassifierStep(Step):
    """The SelectSamplerClassifierStep entity"""

    step_name = "Select Sampler and Classifier"
    step_description = "Trains combinations of samplers and classifiers and selects the best one"

    def __init__(self, sampling_methods: List[SamplingMethodEnum],
                 classification_algorithms: List[ClassificationAlgorithmEnum],
                 scaled_X_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 scaled_X_validation: pd.DataFrame,
                 y_validation: pd.DataFrame):
        """
        Class constructor
        :param sampling_methods: the sampling methods that should be trained and validated
        :param classification_algorithms: the classification algorithms that should be trained and validated
        :param scaled_X_train: scaled dataset with attributes for training
        :param y_train: targets for training
        :param scaled_X_validation: scaled dataset with attributes for validation
        :param y_validation: targets for validation
        """
        self.step_input = SelectSamplerClassifierStepInput(
            sampling_methods,
            classification_algorithms,
            scaled_X_train,
            y_train,
            scaled_X_validation,
            y_validation
        )
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""

        sampler, classifier, sampled_scaled_X_train, sampled_y_train, execution_summary = self.__select()

        self.__save_selected_sampler_info(sampler)
        self.__save_selected_classifier_info(classifier)

        self.step_output = SelectSamplerClassifierStepOutput(sampler,
                                                             classifier,
                                                             execution_summary,
                                                             sampled_scaled_X_train,
                                                             sampled_y_train)

    def __select(self) -> Tuple[Sampler, Classifier, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Selects best combination"""
        sampling_methods = self.step_input.sampling_methods
        classification_algorithms = self.step_input.classification_algorithms

        scaled_X_train = self.step_input.scaled_X_train
        y_train = self.step_input.y_train
        scaled_X_validation = self.step_input.scaled_X_validation
        y_validation = self.step_input.y_validation

        scaled_X_validation = pdutils.select_columns(scaled_X_validation, dscfg.COLUMNS_SAMPLING)

        best_classifier = None
        best_sampler = None
        best_sampler_classifier_evaluation_metric = None
        best_sampled_scaled_X_train = None
        best_sampled_y_train = None

        sampling_methods_run = []
        classification_algorithms_run = []
        evaluation_metrics_run = []

        for sampling_method in sampling_methods:
            sampler = sampler_factory.get(sampling_method)
            sampled_scaled_X_train, sampled_y_train = sampler.fit_sample(scaled_X_train, y_train)
            for classification_algorithm in classification_algorithms:
                classifier = classifier_factory.get(classification_algorithm)
                classifier.train(sampled_scaled_X_train, sampled_y_train)
                evaluation_metric = classifier.evaluate(scaled_X_validation, y_validation)
                sampling_methods_run.append(sampling_method)
                classification_algorithms_run.append(classification_algorithm)
                evaluation_metrics_run.append(evaluation_metric)
                if best_sampler_classifier_evaluation_metric is None or evaluation_metric > best_sampler_classifier_evaluation_metric:
                    best_sampler_classifier_evaluation_metric = evaluation_metric
                    best_sampler = sampler
                    best_classifier = classifier
                    best_sampled_scaled_X_train = sampled_scaled_X_train
                    best_sampled_y_train = sampled_y_train

        execution_summary = pd.DataFrame({
            'sampling_method': sampling_methods_run,
            'classification_algorithm': classification_algorithms_run,
            'evaluation_metric': evaluation_metrics_run
        })
        execution_summary.to_csv(cfg.ASSETS_DIR + "select_sampler_classifier_execution_summary.csv")

        return best_sampler, best_classifier, best_sampled_scaled_X_train, best_sampled_y_train, execution_summary

    def __save_selected_classifier_info(self, classifier: Classifier) -> None:
        with open(cfg.ASSETS_DIR + "classifier.txt", 'w') as f:
            f.write("Selected best classifier: ")
            f.write("\n")
            f.write(classifier.algorithm.value)
            f.write("\n")
            f.write("Selected best hyperparameters: ")
            f.write(str(classifier.classifier.get_params()))

    def __save_selected_sampler_info(self, sampler: Sampler) -> None:
        with open(cfg.ASSETS_DIR + "sampler.txt", 'w') as f:
            f.write("Selected best sampler: ")
            f.write("\n")
            f.write(sampler.method.value)


class SelectSamplerClassifierStepInput(StepInput):
    """Input for SelectSamplerClassifierStep"""

    sampling_methods: List[SamplingMethodEnum]
    classification_algorithms: List[ClassificationAlgorithmEnum]
    scaled_X_train: pd.DataFrame
    y_train: pd.DataFrame
    scaled_X_validation: pd.DataFrame
    y_validation: pd.DataFrame

    def __init__(self, sampling_methods: List[SamplingMethodEnum],
                 classification_algorithms: List[ClassificationAlgorithmEnum],
                 scaled_X_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 scaled_X_validation: pd.DataFrame,
                 y_validation: pd.DataFrame):
        """
        Class constructor
        :param sampling_methods: the sampling methods that should be trained and validated
        :param classification_algorithms: the classification algorithms that should be trained and validated
        :param scaled_X_train: scaled dataset with attributes for training
        :param y_train: targets for training
        :param scaled_X_validation: scaled dataset with attributes for validation
        :param y_validation: targets for validation
        """
        self.sampling_methods = sampling_methods
        self.classification_algorithms = classification_algorithms
        self.scaled_X_train = scaled_X_train
        self.y_train = y_train
        self.scaled_X_validation = scaled_X_validation
        self.y_validation = y_validation


class SelectSamplerClassifierStepOutput(StepOutput):
    """Output for SelectSamplerClassifierStep"""

    sampler: Sampler
    classifier: Classifier
    execution_summary: pd.DataFrame
    sampled_scaled_X_train: pd.DataFrame
    sampled_y_train: pd.DataFrame

    def __init__(self,
                 sampler: Sampler,
                 classifier: Classifier,
                 execution_summary: pd.DataFrame,
                 sampled_scaled_X_train: pd.DataFrame,
                 sampled_y_train: pd.DataFrame
                 ):
        """
        Class constructor
        :param sampler: the sampler
        :param classifier: the classifier
        :param execution_summary: the execution summary
        :param sampled_scaled_X_train: the sampled and scaled dataset with attributes for training
        :param sampled_y_train: the sampled dataset with targets for training
        """
        self.sampler = sampler
        self.classifier = classifier
        self.execution_summary = execution_summary
        self.sampled_scaled_X_train = sampled_scaled_X_train
        self.sampled_y_train = sampled_y_train
