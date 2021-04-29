import logging
from typing import Dict, List

import ijson
from drfs import DRPath
from drfs.filesystems import get_fs

from spec2vec_mlops import config

KEYS = config["gnps_json"]["necessary_keys"]


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, file_path: DRPath):
        self.fs = get_fs(str(file_path))
        self.file_path = file_path

    def load_gnps_json(self, ionmode: str = None) -> List[Dict[str, str]]:
        with self.fs.open(self.file_path, "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            results = [{k: item[k] for k in KEYS} for item in items]
            if ionmode:
                results = [
                    item for item in results if item.get("Ion_Mode").lower == ionmode
                ]
        return results
