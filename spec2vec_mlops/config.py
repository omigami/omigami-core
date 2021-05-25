from datetime import datetime
from pathlib import Path

from spec2vec_mlops import config

ROOT_DIR = Path(__file__).parents[0]

SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
NECESSARY_KEYS = config["gnps_json"]["necessary_keys"]

API_SERVER = config["prefect_flow_registration"]["api_server"]
PROJECT_NAME = config["prefect"]["project"]
OUTPUT_DIR = config["prefect"]["output_dir"]
DATASET_FOLDER = config["prefect"]["dataset_folder"]
S3_MODEL_BUCKET = config["prefect"]["s3_model_bucket"]

MODEL_DIR = config["mlflow"]["model_folder"]
MLFLOW_SERVER = config["mlflow"]["url"]["remote"]
CUSTOM_RESOURCE_INFO = config["k8s"]["custom_seldon_resource"]

# REDIS SETTINGS
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = config["redis"]["spectrum_id_sorted_set"]
SPECTRUM_HASHES = config["redis"]["spectrum_hashes"]
DOCUMENT_HASHES = config["redis"]["document_hashes"]
EMBEDDING_HASHES = config["redis"]["embedding_hashes"]

RedisDBDatasetSize = {"small": "2", "10k": "1", "full": "0"}
DATASET_DIR = {
    "small": f"spec2vec-training-flow/downloaded_datasets/small/{datetime.now().date()}/",
    "10k": f"spec2vec-training-flow/downloaded_datasets/test_10k/",
    "full": f"spec2vec-training-flow/downloaded_datasets/full/2021-05-14/",
}
