from pathlib import Path

from spec2vec_mlops import config

SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
API_SERVER = config["prefect_flow_registration"]["api_server"]
PROJECT_NAME = config["prefect"]["project"]
OUTPUT_DIR = config["prefect"]["output_dir"]
DATASET_FOLDER = config["prefect"]["dataset_folder"]
MODEL_DIR = config["mlflow"]["model_folder"]
MLFLOW_SERVER = config["mlflow"]["url"]["remote"]
ROOT_DIR = Path(__file__).parents[0]
