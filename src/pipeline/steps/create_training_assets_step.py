"""
Module which contains the CreateTrainingAssetsStep, CreateTrainingAssetsStepInput and CreateTrainingAssetsStepOutput classes
They contain the required methods to generate the inputs that are required for machine learning
"""
from typing import List

import pandas as pd

import sklearn

import config.data_preparation_settings as dpcfg
import config.dataset_settings as dscfg
import config.general_settings as cfg

import src.utils.pandas_utils as pdutils
from src.pipeline.step import Step, StepInput, StepOutput


class CreateTrainingAssetsStep(Step):
    """The CreateTrainingAssetsStep entity"""
    step_name = "Create Training Assets"
    step_description = "Create the training assets to be used later on"

    def __init__(self, input_dataset: pd.DataFrame):
        """
        Class constructor
        :param input_dataset: the input dataset
        """
        self.step_input = CreateTrainingAssetsStepInput(input_dataset)
        self.prepare()

    def run(self) -> None:
        """Internal run for step"""
        input_dataset = self.step_input.input_dataset

        target_column_names = dscfg.COLUMNS_TARGET_FOR_ML

        X = pdutils.delete_columns(input_dataset, target_column_names)
        y = pdutils.select_columns(input_dataset, target_column_names)

        X_selected, X_test, y_selected, y_test = self.__split_dataset(X, y)
        X_train, X_validation, y_train, y_validation = self.__split_dataset(X_selected, y_selected)

        self.step_output = CreateTrainingAssetsStepOutput(X_train, y_train,
                                                          X_validation, y_validation,
                                                          X_test, y_test)

    def __split_dataset(self, X: pd.DataFrame, y: pd.DataFrame) -> List[pd.DataFrame]:
        """
        To split the datasets into train and test
        :param X: attribute space
        :param y: target space
        :return: The splits
        """
        return sklearn.model_selection.train_test_split(X, y, test_size=(1 - dpcfg.TRAIN_SIZE), shuffle=False, random_state=cfg.SEED)


class CreateTrainingAssetsStepInput(StepInput):
    """Input for CreateTrainingAssetsStep"""

    input_dataset: pd.DataFrame

    def __init__(self, input_dataset: pd.DataFrame):
        """
        Class constructor
        :param input_dataset: the input dataset
        """
        self.input_dataset = input_dataset


class CreateTrainingAssetsStepOutput(StepOutput):
    """Output for CreateTrainingAssetsStep"""

    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    X_train_validation: pd.DataFrame
    y_train_validation: pd.DataFrame

    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 X_validation: pd.DataFrame,
                 y_validation: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_test: pd.DataFrame):
        """
        Class constructor
        :param X_train: attributes for training
        :param y_train: targets for training
        :param X_validation: attributes for validation
        :param y_validation: targets for validation
        :param X_test: attributes for testing
        :param y_test: targets for testing
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_validation = pdutils.join_dataframes_x_wise(
            [X_train, X_validation]
        )
        self.y_train_validation = pdutils.join_dataframes_x_wise(
            [y_train, y_validation]
        )
