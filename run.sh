#!/bin/bash
set -e

BASE_DIR=$(dirname $0)
PYTEST_CONFIG="${BASE_DIR}/config/.coveragerc"
FLAKE8_CONFIG="${BASE_DIR}/config/.flake8"
PYTHONPATH="${BASE_DIR}"

run_directories_check()
{
  mkdir -p output/
  mkdir -p output/execution_objects/
  mkdir -p output/figures/
  mkdir -p output/deployed_model/
  mkdir -p output/predictions/
  mkdir -p datasets/
  mkdir -p datasets/generated/
  mkdir -p assets/
}

run_flake8()
{
  flake8 . --config=$FLAKE8_CONFIG
}

run_pytest()
{
  pytest --cov-config=$PYTEST_CONFIG --cov=. test/ --cov-fail-under=75 --cov-report=html
}

run_fit()
{
  TF_CPP_MIN_LOG_LEVEL=3 python3 -W ignore PantanalFireDetection.py fit
}

run_predict()
{
  TF_CPP_MIN_LOG_LEVEL=3 python3 -W ignore PantanalFireDetection.py predict
}

if [ "$1" == "build" ]; then
  echo ">>>>>>>>>>>> [1/3] Running directories check"
  run_directories_check
  echo ">>>>>>>>>>>> [2/3] Running flake8"
  run_flake8
  echo ">>>>>>>>>>>> [3/3] Running pytest"
  run_pytest
elif [ "$1" == "fit" ]; then
  echo ">>>>>>>>>>>> [1/2] Running directories check"
  run_directories_check
  echo ">>>>>>>>>>>> [2/2] Fitting the model"
  run_fit
elif [ "$1" == "predict" ]; then
  echo ">>>>>>>>>>>> [1/2] Running directories check"
  run_directories_check
  echo ">>>>>>>>>>>> [2/2] Making predictions"
  run_predict
elif [ "$1" == "all" ]; then
  echo ">>>>>>>>>>>> [1/5] Running directories check"
  run_directories_check
  echo ">>>>>>>>>>>> [2/5] Running flake8"
  run_flake8
  echo ">>>>>>>>>>>> [3/5] Running pytest"
  run_pytest
  echo ">>>>>>>>>>>> [4/5] Fitting the model"
  run_fit
  echo ">>>>>>>>>>>> [5/5] Making predictions"
  run_predict
else
  echo ">>>>>>>>>>>> Option \"$1\" not found. Please try again with one of the following options:"
  echo "- build - fit - predict - all"
fi