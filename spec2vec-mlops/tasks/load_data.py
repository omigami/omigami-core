from typing import Dict
from urllib.request import urlopen

import ijson


class DataLoader:
    def __init__(self):
        pass

    def load_gnps_json(self, uri: str) -> Dict:
        uri = urlopen(uri)
        items = ijson.items(uri, "item")
        objects = ({k: item[k] for k in item.keys()} for item in items)
        for object in objects:
            yield object
