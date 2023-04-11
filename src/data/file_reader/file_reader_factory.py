"""Module which represents a factory for FileReader"""

from src.data.file_reader.csv_file_reader import CSVFileReader
from src.data.file_reader.file_reader import FileReader
from src.data.file_reader.xlsx_file_reader import XLSXFileReader
from src.enum.file_extensions_enum import FileExtensionEnum
from src.exception.not_implemented_exception import NotImplementedException


def get(file_extension: FileExtensionEnum) -> FileReader:
    """
    Factory method for FileReaders
    :param file_extension: The file extension of the file (".csv, ".txt") but in
    FileExtensionEnum type
    :return: The specific file reader for such file extension
    """
    if not file_extension:
        raise ValueError("Parameter file_extension must not be null")
    if not isinstance(file_extension, FileExtensionEnum):
        raise TypeError("Parameter file_extension must be of type FileExtensionEnum")
    if file_extension == FileExtensionEnum.CSV:
        return CSVFileReader()
    if file_extension == FileExtensionEnum.XLSX:
        return XLSXFileReader()
    raise NotImplementedException("No FileReader implemented for extension {}"
                                  .format(file_extension.value))
