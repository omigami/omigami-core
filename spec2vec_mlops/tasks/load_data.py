import logging
from typing import Dict
from urllib.request import urlopen

import ijson

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        pass

    def load_gnps_json(self, uri: str) -> Dict:
        uri = urlopen(uri)
        logger.info(f"Loading data from {uri}... This might take a while.")
        items = ijson.items(uri, "item", multiple_values=True)
        for item in items:
            yield item
