"""Module representing the program menu"""
from src.enum.menu_options_enum import MenuOptionEnum
from src.pipeline import pipeline_executor


def execute(argv: str) -> None:
    """
    Executes the program according to the option chosen
    :param argv: the user inputs
    """
    arg = argv[0]
    if arg == MenuOptionEnum.FIT.value:
        pipeline_executor.execute_fit_pipeline()
    elif arg == MenuOptionEnum.PREDICT.value:
        pipeline_executor.execute_predict_pipeline()
    else:
        raise Exception("Menu option {} not implemented".format(argv[0]))
