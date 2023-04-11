from unittest import mock

import errno
import pytest

from src.utils import os_utils

DIR_PATH = "dummy"
FILE_PATH = "dir/file1"
LIST_OF_FILES = [FILE_PATH, "dir2/file2", "dir2/innerdir/file3"]


@mock.patch('os.remove')
def test_silent_remove(mock_os_remove):
    os_utils.silent_remove(FILE_PATH)

    assert mock_os_remove.call_count == 1


@mock.patch('os.remove')
def test_silent_remove_error_enoent(mock_os_remove):
    exception = OSError()
    exception.errno = errno.ENOENT
    mock_os_remove.side_effect = exception
    os_utils.silent_remove(FILE_PATH)

    assert mock_os_remove.call_count == 1


@mock.patch('os.remove')
def test_silent_remove_error_other(mock_os_remove):
    exception = OSError()
    exception.errno = errno.EPIPE
    mock_os_remove.side_effect = exception

    with pytest.raises(OSError) as e_info:
        os_utils.silent_remove(FILE_PATH)
    assert e_info.value.errno == exception.errno


@mock.patch('os.remove')
@mock.patch('os.listdir')
def test_silent_clean_dir(mock_os_listdir, mock_os_remove):
    mock_os_listdir.return_value = LIST_OF_FILES
    os_utils.silent_clean_dir(DIR_PATH)

    assert mock_os_remove.call_count == len(LIST_OF_FILES)


@mock.patch('os.path.isfile')
def test_check_file_exists(mock_os_path_is_file):
    mock_os_path_is_file.return_value = True
    isfile = os_utils.check_file_exists(FILE_PATH)
    mock_os_path_is_file.return_value = False
    isfile_2 = os_utils.check_file_exists(FILE_PATH)

    assert isfile
    assert not isfile_2


@mock.patch('os.path.isfile')
@mock.patch('shutil.copy')
def test_copy_file_to_path(mock_shutil_copy, mock_os_path_is_file):
    mock_os_path_is_file.return_value = True
    orig_path = "orig_path/" + FILE_PATH
    dst_path = "dst_path/" + FILE_PATH
    os_utils.copy_file_to_path(orig_path, dst_path)
    mock_os_path_is_file.assert_called_with(orig_path)
    mock_shutil_copy.assert_called_with(orig_path, dst_path)

    mock_os_path_is_file.return_value = False
    os_utils.copy_file_to_path(orig_path, dst_path)
    mock_os_path_is_file.assert_called_with(orig_path)

    assert mock_os_path_is_file.call_count == 2
    assert mock_shutil_copy.call_count == 1