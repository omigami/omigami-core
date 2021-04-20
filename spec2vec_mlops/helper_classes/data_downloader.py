import logging
from datetime import datetime
from uuid import uuid4

import requests
from drfs import DRPath
from drfs.filesystems import get_fs

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataDownloader:
    def __init__(self, out_dir: DRPath):
        self.fs = get_fs(str(out_dir))
        date = datetime.today().strftime("%Y-%m-%d")
        self.out_dir = out_dir / date

    def download_gnps_json(
        self,
        uri: str,
    ) -> str:
        if self.fs.exists(self.out_dir):
            files = self.fs.ls(self.out_dir)
            if len(files) != 1:
                raise Exception
            logger.info(f"Using previously downloaded data")
            file_path = str(files[0])
        else:
            file_path = self._download_and_serialize(uri)
        return file_path

    def _download_and_serialize(self, uri: str) -> str:
        """solution is from https://stackoverflow.com/a/16696317/15485553"""
        file_path = self._make_path()
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

    def _make_path(self) -> str:
        file_id = str(uuid4())
        path = str(self.out_dir / f"{file_id}.json")
        logger.info(f"Writing file to {path}")
        return path
