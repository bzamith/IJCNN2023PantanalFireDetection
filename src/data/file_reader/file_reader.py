"""Module which contains the FileReader class"""
import pathlib

import pandas as pd

from src.enum.file_extensions_enum import FileExtensionEnum


class FileReader:
    """
    The FileReader abstract entity
    This is the abstract class
    Specific file extensions readers should extend this
    """

    file_extension: FileExtensionEnum

    def read_data(self, full_file_path: str) -> pd.DataFrame:
        """
        Performs common checks for file readers
        This method is from a abstract class and should be called inside a concrete class
        :param full_file_path: The path on which the file can be found
        :return: A dataframe with the data from file
        """
        if self.__class__ == FileReader:
            raise Exception("Class FileReader must not be called directly")
        if not full_file_path:
            raise ValueError("Parameter full_file_path must not be null")
        file_extension = pathlib.Path(full_file_path).suffix
        if not file_extension:
            raise ValueError("File extension not identified for file from path: {}"
                             .format(full_file_path))
        if file_extension != self.file_extension.value:
            raise ValueError("File extension {} doesn't match reader extension {}"
                             .format(file_extension, self.file_extension.value))
        return pd.DataFrame()
