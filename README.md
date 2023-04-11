# Pantanal Fire Detection

## About

- Authors: Bruna Zamith Santos (programmer), Ricardo Cerri, Marcelo Narciso, Balbina Soriano, Diego Furtado
- This is the codebase for the paper "A New Time Series Framework for Forest Fire Risk Forecasting and Classification" - IJCNN 2023.


## Install

### Using virtualenv (PREFERRED)

```
virtualenv -p /usr/bin/python3 env
source env/bin/activate
pip3 install -r requirements.txt
deactivate
```

### Local

```
sudo python3 setup.py clean --all install
```

## Data for Fitting the Model

There are 2 required data files:

- One with climatic data. Must contain the columns Year, Month, Day, T (temperature), P (precipitation), UR (relative humidity) and V (wind speed).
- One with hotspot detection. Must contain the column Date, representation the dates where a hotspot was identified.

To define which data file will be used, you must:

1. Place the data file in folder under `/datasets` that represents this data
   source. Either `/datasets/hotspot_data` or `/datasets/climatic_data`
2. Define the file names in `/config/general_settings.py`

## Data for Predictions

There is 1 required data file:

- One with climatic data, with at least `X` days, in which `X` is equal to `OBSERVATION_WINDOW` in `/config/forecast_settings.py`

To define which data file will be used, you must:

1. Place the data file in folder under `/datasets/prediction_data`
2. Define the file name in `/config/general_settings.py`

## Settings

You can provide custom settings for all files in `/config` folder

## Run

```
chmod +x run.sh

# Don't forget to activate the virtualenv, if you are using one!
# First, build the code:
./run.sh build

# Then, train the different models:
./run.sh fit

# Finally, predict the risk rate:
./run.sh predict

# To run everything at once, just run:
./run.sh all
```
