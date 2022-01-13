from pathlib import Path

import confuse


class Configuration(confuse.Configuration):
    def config_dir(self):
        return "omigami/ms2deepscore"


config = Configuration("omigami/ms2deepscore", __name__)
DIRECTORIES = config["storage"]["directory"].get(dict)
BINNED_SPECTRUM_HASHES = config["storage"]["redis"]["binned_spectrum_hashes"].get(str)
PROJECT_NAME = config["prefect"]["project"].get(str)
SPECTRUM_IDS_CHUNK_SIZE = 10000
PREDICTOR_ENV_PATH = Path(__file__).parent / "predictor_env.yaml"
