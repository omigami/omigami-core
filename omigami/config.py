import datetime
import os
from pathlib import Path

import confuse
from drfs import DRPath
from typing_extensions import Literal

config = confuse.Configuration("omigami", __name__)
ENV = os.getenv("OMIGAMI_ENV", "local")

ROOT_DIR = Path(__file__).parents[0]

# Prefect
API_SERVER_URLS = config["prefect"]["api_server"].get(dict)

MLFLOW_SERVER = config["mlflow"].get(str)
SELDON_PARAMS = config["seldon"].get(dict)

# Storage
if ENV == "local":
    STORAGE_ROOT = Path(__file__).parent.parent / "local-prefect"
else:
    STORAGE_ROOT = DRPath(config["storage"]["root"][ENV])

REDIS_DATABASES = {  # I think we don't need env differentiation here
    "local": {"small": "2", "10k": "1", "complete": "0"},
    "dev": {"small": "2", "10k": "1", "complete": "0"},
    "prod": {"small": "2", "complete": "0"},
}[ENV]

SOURCE_URI_COMPLETE_GNPS = config["gnps_uri"]["complete"].get(str)
SOURCE_URI_PARTIAL_GNPS = config["gnps_uri"]["partial"].get(str)
SOURCE_URI_PARTIAL_GNPS_500_SPECTRA = config["gnps_uri"]["partial_500_spectra"].get(str)

CLUSTERS = config["clusters"].get(dict)
ION_MODES = {"positive", "negative"}
IonModes = Literal["positive", "negative"]


DEFAULT_PREFECT_TASK_CONFIG = dict(
    max_retries=3, retry_delay=datetime.timedelta(seconds=10)
)

DATASET_IDS = config["storage"]["dataset_dir"].get(dict)
