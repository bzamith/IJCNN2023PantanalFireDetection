"""
Module which contains the CSVFileReader class
It contains the required methods to extract data
from a csv file into a pandas.DataFrame
"""

import pandas as pd

from src.data.file_reader.file_reader import FileReader
from src.enum.file_extensions_enum import FileExtensionEnum


class CSVFileReader(FileReader):
    """The CSVFileReader entity"""

    file_extension = FileExtensionEnum.CSV

    def read_data(self, full_file_path: str) -> pd.DataFrame:
        """
        Reads the data from datafile
        :param full_file_path: The path on which the file can be found
        :return: A dataframe with the data from file
        """
        super(CSVFileReader, self).read_data(full_file_path)
        return pd.read_csv(full_file_path)
