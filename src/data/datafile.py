"""
Module which contains the Datafile class
Datafile is a representation of a file plus the data that it stores
"""
import pathlib

import pandas as pd

import config.general_settings as cfg

from src.data.file_reader import file_reader_factory
from src.enum.data_sources_enum import DataSourceEnum
from src.enum.file_extensions_enum import FileExtensionEnum


class Datafile:
    """
    The Datafile entity
    File + data
    """

    data_source: DataSourceEnum
    file_name: str
    full_file_path: str
    file_extension: FileExtensionEnum
    data: pd.DataFrame

    def __init__(self, data_source: DataSourceEnum, file_name: str):
        """
        Class constructor
        :param data_source: The source of the data for this object
        :param file_name: The name of the file that contains the data
        """
        if not data_source:
            raise ValueError("Parameter data_source must not be null")
        if not isinstance(data_source, DataSourceEnum):
            raise TypeError("Parameter data_source must be of type DataSourceEnum")
        if not file_name:
            raise ValueError("Parameter file_name must not be null")

        self.data_source = data_source
        self.file_name = file_name
        self.__extract_full_file_path()
        self.__extract_file_extension()
        self.__extract_data()

    def __extract_full_file_path(self) -> None:
        """Extract full file path from file name and data source"""
        self.full_file_path = "{data_dir}{data_source_name}/{file_name}".format(data_dir=cfg.DATA_DIR,
                                                                                data_source_name=self.data_source.value,
                                                                                file_name=self.file_name)

    def __extract_file_extension(self) -> None:
        """Extract file extension from file name, so that the correct file reader is set"""
        file_extension_name = pathlib.Path(self.file_name).suffix
        if not file_extension_name:
            raise ValueError("File extension not found: {}".format(self.file_name))
        try:
            self.file_extension = FileExtensionEnum(file_extension_name)
        except ValueError:
            raise ValueError("File extension not recognized: {}".format(file_extension_name))

    def __extract_data(self) -> None:
        """Extract data from file"""
        file_reader = file_reader_factory.get(self.file_extension)
        self.data = file_reader.read_data(self.full_file_path)
