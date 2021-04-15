import logging
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import ijson
import requests
from drfs.filesystems import get_fs

from spec2vec_mlops import config

KEYS = config["gnps_json"]["necessary_keys"].get(list)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, out_dir: Path):
        self.fs = get_fs(str(out_dir))
        self.out_dir = out_dir

    def load_gnps_json(
        self,
        uri: str,
    ) -> List[Dict[str, str]]:
        in_file = self._download_and_serialize(uri)
        return self.parse_json(in_file)

    def parse_json(self, path: str) -> List[Dict[str, str]]:
        with self.fs.open(path, "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            results = [{k: item[k] for k in KEYS} for item in items]
        return results

    def _download_and_serialize(self, uri: str) -> str:
        """solution is from https://stackoverflow.com/a/16696317/15485553"""
        file_path: str = self._make_path()
        logger.info(f"Loading data from {uri}... This might take a while.")
        try:
            with requests.get(uri, stream=True) as r:
                r.raise_for_status()
                with self.fs.open(file_path, "wb") as f:
                    for chunk in r.iter_content(
                        chunk_size=10 * 1024 * 1024
                    ):  # 10 MBs of chunks
                        f.write(chunk)
                        file_size = Path(file_path).stat().st_size
        except requests.ConnectionError:
            self._resume_download(file_size, uri, file_path)
        except requests.Timeout:
            self._resume_download(file_size, uri, file_path)
        except requests.HTTPError:
            self._resume_download(file_size, uri, file_path)
        except requests.TooManyRedirects:
            self._resume_download(file_size, uri, file_path)
        return file_path

    def _resume_download(self, file_size: int, uri: str, path: str):
        """solution is from https://stackoverflow.com/questions/22894211/how-to-resume-file-download-in-python """
        resume_header = {"Range": f"bytes={file_size}-"}
        with requests.get(uri, stream=True, headers=resume_header) as r:
            r.raise_for_status()
            with self.fs.open(path, "ab") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)

    def _make_path(self) -> str:
        file_id = str(uuid4())
        path = f"{str(self.out_dir)}/{file_id}.json"
        return path
