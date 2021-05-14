import os
from pathlib import Path

import yaml

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

with open(
    os.path.join(os.path.dirname(__file__), "config_default.yaml")
) as yaml_config_file:
    config = yaml.safe_load(yaml_config_file)

SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
API_SERVER = config["prefect_flow_registration"]["api_server"]
PROJECT_NAME = config["prefect"]["project"]
OUTPUT_DIR = config["prefect"]["output_dir"]
DATASET_FOLDER = config["prefect"]["dataset_folder"]
MODEL_DIR = config["mlflow"]["model_folder"]
MLFLOW_SERVER = config["mlflow"]["url"]["remote"]
ROOT_DIR = Path(__file__).parents[0]
