import logging
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import ijson
import requests
from drfs.filesystems.base import FileSystemBase
from drfs.filesystems.local import LocalFileSystem
from drfs.filesystems.s3 import S3FileSystem
from drfs.path import RemotePath

from spec2vec_mlops import config

KEYS = config["gnps_json"]["necessary_keys"].get(list)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        remote_fs: Optional[FileSystemBase] = None,
        local_fs: Optional[FileSystemBase] = None,
    ):
        self.remote_fs = remote_fs or S3FileSystem()
        self.local_fs = local_fs or LocalFileSystem()

    def load_gnps_json(
        self, uri: str, out_dir: Path
    ) -> List[Dict[str, str]]:
        fs = self.remote_fs if out_dir is None else self.local_fs
        in_file: str = self._download_and_serialize(uri, fs, out_dir)
        return self.parse_json(fs, in_file)

    @staticmethod
    def parse_json(fs: FileSystemBase, path: str) -> List[Dict[str, str]]:
        with fs.open(path, "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            results = [{k: item[k] for k in KEYS} for item in items]
        return results

    def _download_and_serialize(
        self,
        uri: str,
        fs: FileSystemBase,
        out_dir: Optional[Path],
    ) -> str:
        """solution is from https://stackoverflow.com/a/16696317/15485553"""
        file_path: str = self._make_path(out_dir)
        logger.info(f"Loading data from {uri}... This might take a while.")
        try:
            with requests.get(uri, stream=True) as r:
                r.raise_for_status()
                with fs.open(file_path, "wb") as f:
                    for chunk in r.iter_content(
                        chunk_size=10 * 1024 * 1024
                    ):  # 10 MBs of chunks
                        f.write(chunk)
                        file_size = Path(file_path).stat().st_size
        except requests.ConnectionError:
            self._resume_download(file_size, uri, fs, file_path)
        except requests.Timeout:
            self._resume_download(file_size, uri, fs, file_path)
        except requests.HTTPError:
            self._resume_download(file_size, uri, fs, file_path)
        except requests.TooManyRedirects:
            self._resume_download(file_size, uri, fs, file_path)
        return file_path

    @staticmethod
    def _resume_download(file_size: bytes, uri: str, fs: FileSystemBase, path: str):
        """solution is from https://stackoverflow.com/questions/22894211/how-to-resume-file-download-in-python """
        resume_header = {"Range": f"bytes={file_size}-"}
        with requests.get(uri, stream=True, headers=resume_header) as r:
            r.raise_for_status()
            with fs.open(path, "ab") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB of chunks
                    f.write(chunk)

    @staticmethod
    def _make_path(out_dir: Path) -> str:
        if out_dir is None:
            out_dir = (
                RemotePath("s3://dr-prefect/spec2vec-data")
                # TODO: change this to prod bucket
            )
        file_id = str(uuid4())
        path = f"{str(out_dir)}/{file_id}.json"
        return path
