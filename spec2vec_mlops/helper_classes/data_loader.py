import logging
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen
from uuid import uuid4

import ijson
import requests
from drfs.filesystems import get_fs

from spec2vec_mlops import config

KEYS = config["gnps_json"]["necessary_keys"].get(list)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def load_gnps_json(self, uri: str) -> List[Dict]:
        logger.info(f"Loading data from {uri}... This might take a while.")
        response = urlopen(uri)
        items = ijson.items(response, "item", multiple_values=True)
        results = [{k: item[k] for k in KEYS} for item in items]
        return results

    @staticmethod
    def _save(uri: str, out_dir: Path = None):
        # solution is from https://stackoverflow.com/a/16696317/15485553
        if out_dir is None:
            out_dir = (
                "s3://dr-prefect/spec2vec-data/"  # TODO: change this to prod bucket
            )
        file_id = str(uuid4())
        filename = f"{str(out_dir)}/{file_id}.json"
        logger.info(f"Loading data from {uri}... This might take a while.")
        with requests.get(uri, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=536870912):  # 512 MBs of chunks
                    f.write(chunk)
        return filename

    def load(self, uri: str, out_dir: Path = None):
        in_file = self._save(uri, out_dir)
        fs = get_fs(in_file)
        with fs.open(in_file, "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            results = [{k: item[k] for k in KEYS} for item in items]
        return results
