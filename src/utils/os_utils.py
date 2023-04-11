"""Module with utilities for handling OS operations (specially directories operations)"""
import errno
import os
import shutil


def copy_file_to_path(orig_file_path: str, dst_file_path: str) -> None:
    """
    Check if file exists and, if so, copies it to another path.
    :param orig_file_path: origin file path
    :param dst_file_path: destination file path
    """
    if check_file_exists(orig_file_path):
        shutil.copy(orig_file_path, dst_file_path)


def silent_remove(file_path: str) -> None:
    """
    Check if file exists and, if so, removes it. Otherwise, ignores.
    :param file_path: the file path to be removed
    """
    try:
        os.remove(file_path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise e


def silent_clean_dir(dir_path: str) -> None:
    """
    Silently remove all files of a dir.
    :param dir_path: the directory to be cleaned
    """
    for filename in os.listdir(dir_path):
        silent_remove(os.path.join(dir_path, filename))


def check_file_exists(file_path: str) -> bool:
    """
    Check if file exists
    :param file_path: the file path
    :return: whether the file exists or not
    """
    return os.path.isfile(file_path)
