from datetime import datetime
from pathlib import Path

from spec2vec_mlops import default_configs

ROOT_DIR = Path(__file__).parents[0]

SOURCE_URI_COMPLETE_GNPS = default_configs["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = default_configs["gnps_json"]["uri"]["partial"]
NECESSARY_KEYS = default_configs["gnps_json"]["necessary_keys"]

API_SERVER = default_configs["prefect_flow_registration"]["api_server"]
PROJECT_NAME = default_configs["prefect"]["project"]
OUTPUT_DIR = default_configs["prefect"]["output_dir"]
DATASET_FOLDER = default_configs["prefect"]["dataset_folder"]
S3_MODEL_BUCKET = default_configs["prefect"]["s3_model_bucket"]

MODEL_DIR = default_configs["mlflow"]["model_folder"]
MLFLOW_SERVER = default_configs["mlflow"]["url"]["remote"]
CUSTOM_RESOURCE_INFO = default_configs["k8s"]["custom_seldon_resource"]

# REDIS SETTINGS
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = default_configs["redis"]["spectrum_id_sorted_set"]
SPECTRUM_HASHES = default_configs["redis"]["spectrum_hashes"]
DOCUMENT_HASHES = default_configs["redis"]["document_hashes"]
EMBEDDING_HASHES = default_configs["redis"]["embedding_hashes"]

RedisDBDatasetSize = {"small": "2", "10k": "1", "full": "0"}
DATASET_DIR = {
    "small": f"spec2vec-training-flow/downloaded_datasets/small/{datetime.now().date()}/",
    "10k": f"spec2vec-training-flow/downloaded_datasets/test_10k/",
    "full": f"spec2vec-training-flow/downloaded_datasets/full/2021-05-14/",
}
