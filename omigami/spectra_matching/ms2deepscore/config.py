import confuse

from omigami.config import STORAGE_ROOT


class Configuration(confuse.Configuration):
    def config_dir(self):
        return "omigami/ms2deepscore"


config = Configuration("omigami/ms2deepscore", __name__)
DIRECTORIES = config["storage"]["directory"].get(dict)
BINNED_SPECTRUM_HASHES = config["storage"]["redis"]["binned_spectrum_hashes"].get(str)
PROJECT_NAME = config["prefect"]["project"].get(str)
MS2DEEPSCORE_ROOT = STORAGE_ROOT / "ms2deepscore"
SPECTRUM_IDS_CHUNK_SIZE = 10000
