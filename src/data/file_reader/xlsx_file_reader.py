"""
Module which contains the XLSXFileReader class
It contains the required methods to extract data from a xlsx file into a pandas.DataFrame
"""

import pandas as pd

from src.data.file_reader.file_reader import FileReader
from src.enum.file_extensions_enum import FileExtensionEnum


class XLSXFileReader(FileReader):
    """The XLSXFileReader entity"""

    file_extension = FileExtensionEnum.XLSX

    def read_data(self, full_file_path: str) -> pd.DataFrame:
        """
        Reads the data from datafile
        :param full_file_path: The path on which the file can be found
        :return: A dataframe with the data from file
        """
        super(XLSXFileReader, self).read_data(full_file_path)

        all_sheets = pd.read_excel(full_file_path, sheet_name=None)
        for key in all_sheets:
            return all_sheets[key]

    def read_data_with_sheet_name(self, full_file_path: str, sheet_name: str) -> pd.DataFrame:
        """
        Reads the data from datafile, but for given sheet_name
        :param full_file_path: The path on which the file can be found
        :param sheet_name: The specific sheet from which the data will be extracted
        :return: A dataframe with the data from file
        """
        super(XLSXFileReader, self).read_data(full_file_path)
        return pd.read_excel(full_file_path, sheet_name=sheet_name)
