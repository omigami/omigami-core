from pathlib import Path

import confuse

from omigami.config import STORAGE_ROOT


class Configuration(confuse.Configuration):
    def config_dir(self):
        return "omigami/spec2vec"


config = Configuration("omigami/spec2vec", __name__)
SPEC2VEC_ROOT = STORAGE_ROOT / "spec2vec"

DOCUMENT_DIRECTORIES = config["storage"]["documents"].get(dict)

DOCUMENT_HASHES = config["storage"]["redis"]["document_hashes"].get(str)

REDIS_HOST = config["storage"]["redis"]["env_vars"]["redis_host"].get(str)
REDIS_DB = config["storage"]["redis"]["env_vars"]["redis_db"].get(str)
PREDICTOR_ENV_PATH = Path(__file__).parent / "predictor_env.yaml"
