"""Module with utilities for logging operations"""
import logging

import config.general_settings as cfg

from src.utils import os_utils


def print_and_log(message: str) -> None:
    """
    Print to console and log message.
    :param message: the message to be printed and logged
    """
    print(message)
    logging.info(message)


def clear_logs() -> None:
    """
    Clear log file if it exists.
    """
    log_file_name = cfg.LOG_FILE
    if os_utils.check_file_exists(log_file_name):
        open(log_file_name, 'w').close()
