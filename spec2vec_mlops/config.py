import os
from datetime import datetime
from pathlib import Path

import yaml

with open(
    os.path.join(os.path.dirname(__file__), "config_default.yaml")
) as yaml_config_file:
    default_configs = yaml.safe_load(yaml_config_file)

ROOT_DIR = Path(__file__).parents[0]

# Data for download
SOURCE_URI_COMPLETE_GNPS = default_configs["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = default_configs["uri"]["partial"]

# Prefect
API_SERVER = default_configs["prefect"]["api_server"]
PROJECT_NAME = default_configs["prefect"]["project"]

MLFLOW_SERVER = default_configs["mlflow"]
SELDON_PARAMS = default_configs["seldon"]

# Storage
S3_BUCKET = default_configs["storage"]["s3_bucket"]
MODEL_DIR = default_configs["storage"]["model_folder"]
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = default_configs["storage"]["redis"][
    "spectrum_id_sorted_set"
]
SPECTRUM_HASHES = default_configs["storage"]["redis"]["spectrum_hashes"]
DOCUMENT_HASHES = default_configs["storage"]["redis"]["document_hashes"]
EMBEDDING_HASHES = default_configs["storage"]["redis"]["embedding_hashes"]

RedisDBDatasetSize = {"small": "2", "10k": "1", "full": "0"}
DATASET_DIR = {
    "small": f"spec2vec-training-flow/downloaded_datasets/small/{datetime.now().date()}/",
    "10k": f"spec2vec-training-flow/downloaded_datasets/test_10k/",
    "full": f"spec2vec-training-flow/downloaded_datasets/full/2021-05-14/",
}
