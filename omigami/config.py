import datetime
import os
from pathlib import Path
from typing import Dict

import confuse
from drfs import DRPath
from typing_extensions import Literal

config = confuse.Configuration("omigami", __name__)
OMIGAMI_ENV = os.getenv("OMIGAMI_ENV", "local")

ROOT_DIR = Path(__file__).parents[0]

# Prefect
API_SERVER_URLS = config["prefect"]["api_server"].get(dict)

MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", config["mlflow"].get(str))
SELDON_PARAMS = config["seldon"].get(dict)

# Storage
if OMIGAMI_ENV == "local":
    STORAGE_ROOT = Path(__file__).parent.parent / "local-deployment" / "results"
else:
    STORAGE_ROOT = DRPath(config["storage"]["root"][OMIGAMI_ENV].get(str))

MLFLOW_DIRECTORY = STORAGE_ROOT / "mlflow"

REDIS_DATABASES = {
    "local": {"small": "0"},
    "docker": {"small": "0"},
    "dev": {"small": "2", "10k": "1", "complete": "0"},
    "prod": {"small": "2", "complete": "0"},
}[OMIGAMI_ENV]
REDIS_HOST = config["storage"]["redis"]["host"][OMIGAMI_ENV].get(str)
EMBEDDING_HASHES = config["storage"]["redis"]["embedding_hashes"].get(str)
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = config["storage"]["redis"][
    "spectrum_id_sorted_set"
].get(str)
SPECTRUM_HASHES = config["storage"]["redis"]["spectrum_hashes"].get(str)

SOURCE_URI_COMPLETE_GNPS = config["gnps_uri"]["complete"].get(str)
SOURCE_URI_PARTIAL_GNPS = config["gnps_uri"]["partial"].get(str)
SOURCE_URI_PARTIAL_GNPS_500_SPECTRA = config["gnps_uri"]["partial_500_spectra"].get(str)

CLUSTER = config["clusters"].get(dict).get(OMIGAMI_ENV, None)
ION_MODES = {"positive", "negative"}
IonModes = Literal["positive", "negative"]
CONDA_ENV_PATH = Path(__file__).parents[1] / "requirements/environment.frozen.yaml"
CODE_PATH = Path(__file__).parent

DEFAULT_PREFECT_TASK_CONFIG = dict(
    max_retries=3, retry_delay=datetime.timedelta(seconds=10)
)

DATASET_IDS = config["storage"]["dataset_id"].get(dict)


def get_login_config() -> Dict[str, str]:
    if OMIGAMI_ENV in ("dev", "prod"):
        login_config = config["login"][OMIGAMI_ENV].get(dict)
        login_config.pop("token")
    else:
        login_config = {
            "username": None,
            "password": None,
            "auth_url": "url",
            "session_token": "token",
        }
    return login_config


CHUNK_SIZE = int(1e8)
