import datetime
from pathlib import Path

import confuse
from typing_extensions import Literal

config = confuse.Configuration("omigami", __name__)

ROOT_DIR = Path(__file__).parents[0]

# Prefect
API_SERVER_URLS = config["prefect"]["api_server"].get(dict)
PROJECT_NAME = config["prefect"]["project"].get(str)

MLFLOW_SERVER = config["mlflow"].get(str)
SELDON_PARAMS = config["seldon"].get(dict)

# Storage
S3_BUCKETS = config["storage"]["s3_bucket"].get(dict)

REDIS_DATABASES = {
    "dev": {"small": "2", "10k": "1", "complete": "0"},
    "prod": {"complete": "0"},
}


CLUSTERS = config["clusters"].get(dict)
ION_MODES = {"positive", "negative"}
IonModes = Literal["positive", "negative"]


DEFAULT_PREFECT_TASK_CONFIG = dict(
    max_retries=3, retry_delay=datetime.timedelta(seconds=10)
)


def merge_prefect_task_configs(kwargs):
    """helper function to merge DEFAULT CONFIG with kwargs"""
    return {k: v for k, v in {**DEFAULT_PREFECT_TASK_CONFIG.copy(), **kwargs}.items()}
