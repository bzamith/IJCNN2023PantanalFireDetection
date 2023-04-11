"""
Module which contains the ReadPrepareTrainingDataStep, ReadPrepareTrainingDataStepInput and ReadPrepareTrainingDataStepOutput classes
It contains the required methods to perform data preprocessing for the historical data
"""

import pandas as pd

import config.dataset_settings as dscfg
import config.general_settings as cfg

import src.utils.pandas_utils as pdutils
from src.data.datafile import Datafile
from src.enum.data_sources_enum import DataSourceEnum
from src.pipeline.step import Step, StepInput, StepOutput


class ReadPrepareTrainingDataStep(Step):
    """The ReadPrepareTrainingDataStep entity"""

    step_name = "Read and Prepare Historical Data"
    step_description = "Read, prepare and save input dataset from historical data"

    hotspot_file: Datafile
    climatic_data_file: Datafile

    def __init__(self):
        """Class constructor"""
        self.step_input = ReadPrepareTrainingDataStepInput()
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        input_dataset = self.__create_input_dataset()
        input_dataset.to_csv(cfg.DATA_GENERATED_DIR + "fit_input_dataset.csv")

        self.step_output = ReadPrepareTrainingDataStepOutput(input_dataset)

    def __create_input_dataset(self) -> pd.DataFrame:
        """
        Calls different methods sequentially in order to generate the dataset
        :return: the input dataset
        """
        self.__read_files()
        self.__transform_files()
        return self.__merge_files()

    def __read_files(self) -> None:
        """Reads the files into Datafile objects"""
        self.hotspot_file = Datafile(DataSourceEnum.HOTSPOT_DATA, cfg.HOTSPOT_FILE_NAME)
        self.climatic_data_file = Datafile(DataSourceEnum.CLIMATIC_DATA, cfg.CLIMATIC_DATA_FILE_NAME)

    def __transform_files(self) -> None:
        """
        Converts hotspot data 'date' column accordingly

        Ex:
        'Ano' | 'Mês' | 'Dia'
        2002  |  10   |  22

        becomes:
        'Date'
        2002-10-12
        """
        # Hotspot file
        self.__create_date_column_hotspot_file()
        self.hotspot_file.data = pdutils.select_columns(self.hotspot_file.data, dscfg.DATE_COLUMN_NAME).drop_duplicates()

        # Climatic data file
        self.climatic_data_file.data = pdutils.select_columns(self.climatic_data_file.data, dscfg.CLIMATIC_DATA_SELECTED_COLUMNS)

    def __merge_files(self) -> pd.DataFrame:
        """
        Merges the two data
        Left_join on the Date column and creates the 'hotspot_identified' column
        :return: the merged dataframe
        """
        target = dscfg.HOTSPOT_IDENTIFIED_COLUMN_NAME

        dataset = self.climatic_data_file.data.merge(self.hotspot_file.data, on=dscfg.DATE_COLUMN_NAME, how='left', indicator=target)
        dataset = dataset.replace({target: 'left_only'}, 0)
        dataset = dataset.replace({target: 'both'}, 1)
        dataset[target] = pd.to_numeric(dataset[target])

        return dataset

    def __create_date_column_hotspot_file(self) -> None:
        """
        Converts hotspot data 'date' column accordingly

        Ex:
        'Ano' | 'Mês' | 'Dia'
        2002  |  10   |  22

        becomes:
        'Date'
        2002-10-12
        """
        data = self.hotspot_file.data
        self.hotspot_file.data[dscfg.DATE_COLUMN_NAME] = data['Ano'].map(str) + '-' + data['Mês'].map(str) + '-' + data['Dia'].map(str)
        self.hotspot_file.data[dscfg.DATE_COLUMN_NAME] = pd.to_datetime(self.hotspot_file.data[dscfg.DATE_COLUMN_NAME])


class ReadPrepareTrainingDataStepInput(StepInput):
    """Input for ReadPrepareTrainingDataStep"""


class ReadPrepareTrainingDataStepOutput(StepOutput):
    """Output for ReadPrepareTrainingDataStep"""

    dataset: pd.DataFrame

    def __init__(self, dataset: pd.DataFrame):
        """
        Class constructor
        :param dataset: the prepared dataset
        """
        self.dataset = dataset
