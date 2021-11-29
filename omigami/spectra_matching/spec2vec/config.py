import confuse

from omigami.config import STORAGE_ROOT


class Configuration(confuse.Configuration):
    def config_dir(self):
        return "omigami/spec2vec"


config = Configuration("omigami/spec2vec", __name__)
PROJECT_NAME = config["prefect"]["project"].get(str)
SPEC2VEC_ROOT = STORAGE_ROOT / PROJECT_NAME

DOCUMENT_DIRECTORIES = config["storage"]["documents"].get(dict)

DOCUMENT_HASHES = config["storage"]["redis"]["document_hashes"].get(str)

REDIS_HOST = config["storage"]["redis"]["env_vars"]["redis_host"].get(str)
REDIS_DB = config["storage"]["redis"]["env_vars"]["redis_db"].get(str)