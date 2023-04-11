"""
Project: Pantanal Fire Detection
Author: Bruna Zamith Santos
Supervisors: Ricardo Cerri, Marcelo Narciso, Balbina Soriano, Diego Furtado
"""
import logging
import os
import random
import sys

import numpy

import tensorflow

import config.general_settings as cfg

from src.menu import menu
from src.utils import logging_utils

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG,
                    format=cfg.LOG_FORMAT,
                    force=True,
                    filename=cfg.LOG_FILE)

if __name__ == '__main__':
    logging_utils.clear_logs()

    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    random.seed(cfg.SEED)
    numpy.random.seed(cfg.SEED)
    tensorflow.random.set_seed(cfg.SEED)

    menu.execute(sys.argv[1:])
