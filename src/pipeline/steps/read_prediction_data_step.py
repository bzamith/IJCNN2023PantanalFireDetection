"""
Module which contains the ReadPredictionDataStep, ReadPredictionDataStepInput and ReadPredictionDataStepOutput classes
It contains the required methods to perform data preprocessing for the historical data
"""

import pandas as pd

import config.general_settings as cfg

from src.data.datafile import Datafile
from src.enum.data_sources_enum import DataSourceEnum
from src.pipeline.step import Step, StepInput, StepOutput


class ReadPredictionDataStep(Step):
    """The ReadPredictionDataStep entity"""

    step_name = "Read Prediction Data"
    step_description = "Read prediction dataset from prediction data"

    climatic_data_file: Datafile

    def __init__(self):
        """Class constructor"""
        self.step_input = ReadPredictionDataStepInput()
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        dataset = Datafile(DataSourceEnum.PREDICTION_DATA, cfg.PREDICTION_DATA_FILE_NAME).data

        self.step_output = ReadPredictionDataStepOutput(dataset)


class ReadPredictionDataStepInput(StepInput):
    """Input for ReadPredictionDataStep"""


class ReadPredictionDataStepOutput(StepOutput):
    """Output for ReadPredictionDataStep"""

    dataset: pd.DataFrame

    def __init__(self, dataset: pd.DataFrame):
        """
        Class constructor
        :param dataset: the prepared dataset
        """
        self.dataset = dataset
