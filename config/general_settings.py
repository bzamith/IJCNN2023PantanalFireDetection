"""
Config: General Settings File
-----------------------------------
"""
from pathlib import Path

HOTSPOT_FILE_NAME = "DIAS-FOCOS-CALOR_1999-2019_Subregiao-NHECOLANDIA.xlsx"
CLIMATIC_DATA_FILE_NAME = "Climatic_Data_1999_a_2019_SubRegiao-NHECOLANDIA.xlsx"
PREDICTION_DATA_FILE_NAME = "prediction_data.xlsx"

"""
RECOMMENDATION: DO NOT EDIT THE VARIABLES BELOW THIS COMMENT
"""
PROJECT_DIR = str(Path(__file__).resolve().parent.parent) + '/'
BUILD_DIR = "{project_dir}build/".format(project_dir=PROJECT_DIR)
OUTPUT_DIR = "{project_dir}output/".format(project_dir=PROJECT_DIR)
OUTPUT_EXECUTION_OBJECTS_DIR = "{output_dir}execution_objects/".format(output_dir=OUTPUT_DIR)
OUTPUT_DEPLOYED_MODEL_DIR = "{output_dir}deployed_model/".format(output_dir=OUTPUT_DIR)
OUTPUT_PREDICTIONS_DIR = "{output_dir}predictions/".format(output_dir=OUTPUT_DIR)
OUTPUT_FIGURES_DIR = "{output_dir}figures/".format(output_dir=OUTPUT_DIR)
TEST_DIR = "{project_dir}test/".format(project_dir=PROJECT_DIR)
DATA_DIR = "{project_dir}datasets/".format(project_dir=PROJECT_DIR)
ASSETS_DIR = "{project_dir}assets/".format(project_dir=PROJECT_DIR)
DATA_GENERATED_DIR = "{data_dir}generated/".format(data_dir=DATA_DIR)
PREDICTION_DIR = "{data_dir}prediction/".format(data_dir=DATA_DIR)

SCALER_FILE_NAME = "scaler.pkl"
FORECASTER_SUB_DIR = "forecaster_assets"
SAMPLER_FILE_NAME = "sampler.pkl"
CLASSIFIER_FILE_NAME = "classifier.pkl"
DEFINITIONS_DICT_FILE_NAME = "definitions_dict.pkl"

LOG_FILE = "{output_dir}app.log".format(output_dir=OUTPUT_DIR)
LOG_FORMAT = "%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s"

SEED = 0

SCALING_METHOD_KEY = 'scaling_method'
FORECASTING_ALGORITHM_KEY = 'forecasting_algorithm'
SAMPLING_METHOD_KEY = 'sampling_method'
CLASSIFICATION_ALGORITHM_KEY = 'classification_algorithm'
THRESHOLDS_KEY = 'thresholds'
