from pathlib import Path

import confuse
from typing_extensions import Literal

config = confuse.Configuration("omigami", __name__)

ROOT_DIR = Path(__file__).parents[0]

# Data for download
SOURCE_URI_COMPLETE_GNPS = config["gnps_uri"]["complete"].get(str)
SOURCE_URI_PARTIAL_GNPS = config["gnps_uri"]["partial"].get(str)

# Prefect
API_SERVER_URLS = config["prefect"]["api_server"].get(dict)
PROJECT_NAME = config["prefect"]["project"].get(str)

MLFLOW_SERVER = config["mlflow"].get(str)
SELDON_PARAMS = config["seldon"].get(dict)

# Storage
S3_BUCKETS = config["storage"]["s3_bucket"].get(dict)
MODEL_DIRECTORIES = config["storage"]["model_folder"].get(dict)
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = config["storage"]["redis"][
    "spectrum_id_sorted_set"
].get(str)
SPECTRUM_HASHES = config["storage"]["redis"]["spectrum_hashes"].get(str)
DOCUMENT_HASHES = config["storage"]["redis"]["document_hashes"].get(str)
EMBEDDING_HASHES = config["storage"]["redis"]["embedding_hashes"].get(str)
REDIS_DATABASES = {
    "dev": {"small": "2", "10k": "1", "complete": "0"},
    "prod": {"small": "2", "complete": "0"},
}
DATASET_IDS = config["storage"]["dataset_dir"].get(dict)

CLUSTERS = config["clusters"].get(dict)
ION_MODES = {"positive", "negative"}
IonModes = Literal["positive", "negative"]
