import logging
from typing import Dict, List
from urllib.request import urlopen

import ijson

from spec2vec_mlops import config

KEYS = config["gnps_json"]["necessary_keys"]


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def load_gnps_json(self, uri: str) -> List[Dict]:
        uri = urlopen(uri)
        logger.info(f"Loading data from {uri}... This might take a while.")
        items = ijson.items(uri, "item", multiple_values=True)
        results = [{k: item[k] for k in KEYS} for item in items]
        return results
