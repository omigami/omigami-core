import logging
from typing import Dict, List

import ijson
from drfs import DRPath
from drfs.filesystems import get_fs

from spec2vec_mlops import config
from spec2vec_mlops.gateways.redis_gateway import RedisDataGateway

KEYS = config["gnps_json"]["necessary_keys"]


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, file_path: DRPath):
        self.fs = get_fs(str(file_path))
        self.file_path = file_path

    def load_gnps_json(
        self, ionmode: str = None, skip_if_exists: bool = True
    ) -> List[Dict[str, str]]:
        dgw = RedisDataGateway()
        with self.fs.open(self.file_path, "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            results = []
            for item in items:
                dct = {k: item[k] for k in KEYS}
                if dct.get("Ion_Mode").lower() == ionmode:
                    if not skip_if_exists or dgw.list_spectra_not_exist(
                        [dct.get("SpectrumID")]
                    ) == [dct.get("SpectrumID")]:
                        results.append(item)
        return results
