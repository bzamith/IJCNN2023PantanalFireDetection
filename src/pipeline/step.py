"""
Module which contains the Step, StepInput and StepOutput classes
Both are abstract
"""

import logging
import pickle
from typing import Any, List

import config.general_settings as cfg

INPUT_SUFFIX = "_input"
OUTPUT_SUFFIX = "_output"


class Step:
    """The Step entity"""
    step_name: str
    step_description: str
    step_input: Any
    step_output: Any

    def prepare(self) -> None:
        """
        Prepare step to be executed.
        Needs to be called from each implemented step!
        """
        if self.requires_re_run():
            logging.info("Running step {}\n{}".format(
                self.step_name,
                self.step_description
            ))
            self.run()
            self.__save_step_input_object()
            self.__save_step_output_object()
            self.step_output.updated = True
        else:
            logging.info("Skipping step {}".format(
                self.step_name
            ))
            self.step_input = self.__load_step_input_object()
            self.step_output = self.__load_step_output_object()
            self.step_output.updated = False

    def requires_re_run(self) -> bool:
        """
        To determine whether a step needs to be re-run or not
        :return: whether it needs to be re-run or not
        """
        if self.__class__ == Step:
            raise Exception("Class Step must not be called directly")
        return True

    def run(self) -> None:
        """Internal run for step"""
        if self.__class__ == Step:
            raise Exception("Class Step must not be called directly")

    def __save_step_input_object(self) -> None:
        """To save step_input object to file"""
        with open(self.__get_step_input_file_name(), 'wb') as file:
            pickle.dump(self.step_input, file)

    def __save_step_output_object(self) -> None:
        """To save step_output object to file"""
        with open(self.__get_step_output_file_name(), 'wb') as file:
            pickle.dump(self.step_output, file)

    def __load_step_input_object(self) -> Any:
        """
        To load step_input object from file
        :return: the loaded object
        """
        with open(self.__get_step_input_file_name(), 'rb') as file:
            return pickle.load(file)

    def __load_step_output_object(self) -> Any:
        """
        To load step_output object from file
        :return: the loaded object
        """
        with open(self.__get_step_output_file_name(), 'rb') as file:
            return pickle.load(file)

    def __get_step_input_file_name(self) -> str:
        """
        Return the step input file name
        :return: the step input file name
        """
        return cfg.OUTPUT_EXECUTION_OBJECTS_DIR + self.step_name.lower().replace(" ", "_") + INPUT_SUFFIX

    def __get_step_output_file_name(self) -> str:
        """
        Return the step output file name
        :return: the step output file name
        """
        return cfg.OUTPUT_EXECUTION_OBJECTS_DIR + self.step_name.lower().replace(" ", "_") + OUTPUT_SUFFIX


class StepInput:
    """The StepInput entity"""
    inputs: List[Any]


class StepOutput:
    """The StepOutput entity"""
    outputs: List[Any]
    updated = True
