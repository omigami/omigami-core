import datetime
import os
from pathlib import Path
from typing import Dict

import confuse
from drfs import DRPath
from typing_extensions import Literal

from omigami.env_config import (
    Clusters,
    StorageRoots,
    RedisDatabases,
    RedisHosts,
    PrefectServers,
    Environments,
)

ROOT_DIR = Path(__file__).parents[0]
config = confuse.Configuration("omigami", __name__)
OMIGAMI_ENV = os.getenv("OMIGAMI_ENV", Environments.local)

ION_MODES = {"positive", "negative"}
IonModes = Literal["positive", "negative"]

STORAGE_ROOT = DRPath(StorageRoots[OMIGAMI_ENV])
CLUSTER = Clusters[OMIGAMI_ENV]

# Prefect & Tasks
PREFECT_SERVER = PrefectServers[OMIGAMI_ENV]
CHUNK_SIZE = int(1e8)
DATASET_IDS = config["storage"]["dataset_id"].get(dict)
DEFAULT_PREFECT_TASK_CONFIG = dict(
    max_retries=3, retry_delay=datetime.timedelta(seconds=10)
)

# Mlflow
MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", config["mlflow"].get(str))
MLFLOW_DIRECTORY = STORAGE_ROOT / "mlflow"
CONDA_ENV_PATH = ROOT_DIR.parent / "requirements/development/environment.frozen.yaml"
CODE_PATH = ROOT_DIR


SELDON_PARAMS = config["seldon"].get(dict)


# Redis Configurations
REDIS_DATABASES = RedisDatabases[OMIGAMI_ENV]
REDIS_HOST = RedisHosts[OMIGAMI_ENV]
EMBEDDING_HASHES = config["storage"]["redis"]["embedding_hashes"].get(str)
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = config["storage"]["redis"][
    "spectrum_id_sorted_set"
].get(str)
SPECTRUM_HASHES = config["storage"]["redis"]["spectrum_hashes"].get(str)

# URIs for downloading GNPS files
GNPS_URIS = {
    "complete": "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    "small": "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json",
    "small_500": "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS_500_spectra.json",
    "10k": None,  # This file has no source URI and it lives only in S3
}


def get_login_config() -> Dict[str, str]:
    if OMIGAMI_ENV in (Environments.dev, Environments.prod):
        login_config = config["login"][OMIGAMI_ENV].get(dict)
    else:
        login_config = {
            "username": None,
            "password": None,
            "auth_url": "url",
            "session_token": "token",
        }
    return login_config
