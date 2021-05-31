from datetime import datetime
from pathlib import Path

from spec2vec_mlops import ENV

ROOT_DIR = Path(__file__).parents[0]

SOURCE_URI_COMPLETE_GNPS = ENV["gnps_json"]["uri"]["complete"].get()
SOURCE_URI_PARTIAL_GNPS = ENV["gnps_json"]["uri"]["partial"].get()
NECESSARY_KEYS = ENV["gnps_json"]["necessary_keys"].get()

API_SERVER = ENV["prefect_flow_registration"]["api_server"].get()
PROJECT_NAME = ENV["prefect"]["project"].get()
OUTPUT_DIR = ENV["prefect"]["output_dir"].get()
DATASET_FOLDER = ENV["prefect"]["dataset_folder"].get()
S3_MODEL_BUCKET = ENV["prefect"]["s3_model_bucket"].get()

MODEL_DIR = ENV["mlflow"]["model_folder"].get()
MLFLOW_SERVER = ENV["mlflow"]["url"]["remote"].get()
CUSTOM_RESOURCE_INFO = ENV["k8s"]["custom_seldon_resource"].get()

# REDIS SETTINGS
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = ENV["redis"]["spectrum_id_sorted_set"].get()
SPECTRUM_HASHES = ENV["redis"]["spectrum_hashes"].get()
DOCUMENT_HASHES = ENV["redis"]["document_hashes"].get()
EMBEDDING_HASHES = ENV["redis"]["embedding_hashes"].get()

RedisDBDatasetSize = {"small": "2", "10k": "1", "full": "0"}
DATASET_DIR = {
    "small": f"spec2vec-training-flow/downloaded_datasets/small/{datetime.now().date()}/",
    "10k": f"spec2vec-training-flow/downloaded_datasets/test_10k/",
    "full": f"spec2vec-training-flow/downloaded_datasets/full/2021-05-14/",
}
