import datetime
import logging
from typing import Dict, List
from urllib.request import urlopen

import ijson
from prefect import task

from spec2vec_mlops import config

KEYS = config["gnps_json"]["necessary_keys"].get(list)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        pass

    def load_gnps_json(self, uri: str) -> List[Dict]:
        uri = urlopen(uri)
        logger.info(f"Loading data from {uri}... This might take a while.")
        items = ijson.items(uri, "item", multiple_values=True)
        results = [{k: item[k] for k in KEYS} for item in items]
        return results


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def load_data_task(uri) -> List[Dict[str, str]]:
    dl = DataLoader()
    results = dl.load_gnps_json(uri)
    return results
