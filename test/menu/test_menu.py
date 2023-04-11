from unittest import mock

import pytest

from src.menu import menu


@mock.patch('src.pipeline.pipeline_executor.execute_fit_pipeline')
def test_menu_execute_fit(mock_execute_fit_pipeline):
    menu.execute(["fit"])
    mock_execute_fit_pipeline.assert_called_once()


@mock.patch('src.pipeline.pipeline_executor.execute_predict_pipeline')
def test_menu_execute_predict(mock_execute_predict_pipeline):
    menu.execute(["predict"])
    mock_execute_predict_pipeline.assert_called_once()


def test_menu_not_implemented():
    with pytest.raises(Exception) as e_info:
        menu.execute(["dummy"])
    assert str(e_info.value) == "Menu option dummy not implemented"
