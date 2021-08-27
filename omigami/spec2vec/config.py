import confuse


class Configuration(confuse.Configuration):
    def config_dir(self):
        return "omigami/spec2vec"


config = Configuration("omigami/spec2vec", __name__)

MODEL_DIRECTORIES = config["storage"]["model_folder"].get(dict)
DOCUMENT_DIRECTORIES = config["storage"]["documents"].get(dict)

SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = config["storage"]["redis"][
    "spectrum_id_sorted_set"
].get(str)
SPECTRUM_HASHES = config["storage"]["redis"]["spectrum_hashes"].get(str)
DOCUMENT_HASHES = config["storage"]["redis"]["document_hashes"].get(str)
EMBEDDING_HASHES = config["storage"]["redis"]["embedding_hashes"].get(str)
PROJECT_NAME = config["prefect"]["project"].get(str)

REDIS_HOST = config["storage"]["redis"]["env_vars"]["redis_host"].get(str)
REDIS_DB = config["storage"]["redis"]["env_vars"]["redis_db"].get(str)
