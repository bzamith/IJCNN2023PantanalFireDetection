from io import StringIO
from unittest import mock

import typing

import config.general_settings as cfg

import src.utils.logging_utils as lutils

MESSAGE = "dummy"


@mock.patch('logging.info')
@mock.patch('sys.stdout', new_callable=StringIO)
def test_print_and_log(mock_sys_stdout, mock_logging_info):
    lutils.print_and_log(MESSAGE)

    assert mock_sys_stdout.getvalue() == MESSAGE + '\n'
    mock_logging_info.assert_called_once_with(MESSAGE)


@mock.patch('src.utils.os_utils.check_file_exists')
@mock.patch('builtins.open')
def test_clear_logs_file_does_not_exist(mock_open, mock_osutils_check_file_exists):
    mock_osutils_check_file_exists.return_value = False

    lutils.clear_logs()

    assert mock_open.call_count == 0
    mock_osutils_check_file_exists.assert_called_once_with(cfg.LOG_FILE)


@mock.patch('src.utils.os_utils.check_file_exists')
@mock.patch('builtins.open')
@mock.patch('typing.IO.close')
def test_clear_logs_file_exists(mock_close, mock_open, mock_osutils_check_file_exists):
    mock_osutils_check_file_exists.return_value = True
    mock_open.return_value = typing.IO()

    lutils.clear_logs()

    assert mock_open.call_args[0][0] == cfg.LOG_FILE
    assert mock_open.call_args[0][1] == 'w'
    mock_osutils_check_file_exists.assert_called_once_with(cfg.LOG_FILE)
    mock_close.assert_called_once()
