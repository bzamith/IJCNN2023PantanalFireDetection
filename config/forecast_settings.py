"""
Config: Forecasting Settings
-----------------------------------
"""

OBSERVATION_WINDOW = 5
FORECAST_HORIZON = 3
NB_EPOCHS = 250
EARLY_STOPPING_PATIENCE = 10
ERROR_METRIC = "mae"
ALGORITHM = [
    "Long Short Term Memory",
    "Gated Recurrent Unit",
    "Convolutional Neural Network"
]
VERBOSE = True
