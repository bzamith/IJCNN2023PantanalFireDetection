"""
Module which contains the CalculateEvaluationMeasuresStep, CalculateEvaluationMeasuresStepInput and CalculateEvaluationMeasuresStepOutput classes
They contain the required methods to calculate the final measures
"""
import pandas as pd

import config.general_settings as cfg

import src.utils.pandas_utils as pdutils
from src.enum.evaluation_measures_enum import EvaluationMeasureEnum
from src.evaluation.measure.evaluator_factory import get
from src.exception.not_implemented_exception import NotImplementedException
from src.pipeline.step import Step, StepInput, StepOutput


class CalculateEvaluationMeasuresStep(Step):
    """The CalculateEvaluationMeasuresStep entity"""

    step_name = "Calculate Evaluation Measures"
    step_description = "Calculate the final measures to report later on"

    def __init__(self, generated_output_dataset: pd.DataFrame):
        """
        Class constructor
        :param generated_output_dataset: the generated output dataset
        """
        self.step_input = CalculateEvaluationMeasuresStepInput(generated_output_dataset)
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        generated_output_dataset = self.step_input.generated_output_dataset

        evaluation_measures_output = self.__calculate_evaluation_measures(generated_output_dataset)

        evaluation_measures_output.to_csv(cfg.DATA_GENERATED_DIR + "evaluation_measures_dataset.csv")
        self.step_output = CalculateEvaluationMeasuresStepOutput(evaluation_measures_output)

    def __calculate_evaluation_measures(self, input_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the evaluation measures from dataset
        :param input_dataset: the dataset to be used for calculation
        :return: dataset with evaluation measures
        """
        output_dataset = None
        for evaluation_measure in EvaluationMeasureEnum:
            try:
                evaluation_calculator = get(evaluation_measure)
                result = evaluation_calculator.calculate(input_dataset)
                result = pd.DataFrame(result)
                if isinstance(output_dataset, pd.DataFrame):
                    output_dataset = pdutils.join_dataframes_x_wise([output_dataset, result])
                else:
                    output_dataset = result
            except NotImplementedException:
                pass
        return output_dataset


class CalculateEvaluationMeasuresStepInput(StepInput):
    """Input for CalculateEvaluationMeasuresStep"""

    generated_output_dataset: pd.DataFrame

    def __init__(self, generated_output_dataset: pd.DataFrame):
        """
        Class constructor
        :param generated_output_dataset: the generated output dataset
        """
        self.generated_output_dataset = generated_output_dataset


class CalculateEvaluationMeasuresStepOutput(StepOutput):
    """Output for CalculateEvaluationMeasuresStep"""

    dataset: pd.DataFrame

    def __init__(self, dataset: pd.DataFrame):
        """
        Class constructor
        :param dataset: the evaluation measures dataset
        """
        self.dataset = dataset
