import logging
from typing import List, Dict
from uuid import uuid4

import ijson
import requests
from drfs import DRPath
from drfs.filesystems import get_fs

from spec2vec_mlops.tasks.data_gateway import InputDataGateway
from spec2vec_mlops import config

logger = logging.getLogger(__name__)
KEYS = config["gnps_json"]["necessary_keys"]


class FSInputDataGateway(InputDataGateway):
    def __init__(self):
        self.fs = None

    def download_gnps(self, uri: str, dataset_dir: str, dataset_id: str) -> str:
        self.fs = get_fs(dataset_dir)
        dataset_dir = DRPath(dataset_dir) / dataset_id
        if self.fs.exists(dataset_dir):
            files = self.fs.ls(dataset_dir)
            if len(files) > 1:
                raise RuntimeError(
                    f"There are files already present on {dataset_dir}: {files}"
                )
            logger.info(f"Using previously downloaded data")
            file_path = str(files[0])
        else:
            file_path = self._download_and_serialize(uri, dataset_dir)
        return file_path

    def _download_and_serialize(self, uri: str, output_dir: DRPath) -> str:
        """solution is from https://stackoverflow.com/a/16696317/15485553"""
        file_path = self._make_path(output_dir)
        logger.info(f"Loading data from {uri}... This might take a while.")
        chunk_size = 10 * 1024 * 1024
        try:
            with requests.get(uri, stream=True) as r:
                r.raise_for_status()
                with self.fs.open(file_path, "wb") as f:
                    file_size = 0
                    for chunk in r.iter_content(
                        chunk_size=chunk_size
                    ):  # 10 MBs of chunks
                        f.write(chunk)
                        file_size += chunk_size
        except requests.ConnectionError:
            self._resume_download(file_size, uri, file_path)
        except requests.Timeout:
            self._resume_download(file_size, uri, file_path)
        except requests.HTTPError:
            self._resume_download(file_size, uri, file_path)
        except requests.TooManyRedirects:
            self._resume_download(file_size, uri, file_path)
        return file_path

    def _resume_download(self, file_size: int, uri: str, file_path: str):
        """solution is from https://stackoverflow.com/questions/22894211/how-to-resume-file-download-in-python """
        resume_header = {"Range": f"bytes={file_size}-"}
        with requests.get(uri, stream=True, headers=resume_header) as r:
            r.raise_for_status()
            with self.fs.open(file_path, "ab") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)

    @staticmethod
    def _make_path(output_dir: DRPath) -> str:
        file_id = str(uuid4())
        path = str(output_dir / f"{file_id}.json")
        logger.info(f"Writing file to {path}")
        return path

    def load_gnps(self, path: str) -> List[Dict[str, str]]:
        with self.fs.open(DRPath(path), "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            results = [{k: item[k] for k in KEYS} for item in items]
        return results
