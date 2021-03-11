import contextlib
from typing import Dict, List
from urllib.request import urlopen

import ijson


class DataLoader:
    def __init__(self):
        pass

    def load_gnps_json(self, uri: str) -> Dict:
        uri = urlopen(uri)
        items = ijson.items(uri, "item")
        for item in items:
            yield item
