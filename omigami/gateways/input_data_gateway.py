import json
import logging
import pickle
from typing import List, Optional, Any

import ijson
import requests
from drfs import DRPath
from drfs.filesystems import get_fs
from drfs.filesystems.base import FileSystemBase

from omigami.entities.data_models import SpectrumInputData
from omigami.data_gateway import InputDataGateway

logger = logging.getLogger(__name__)
KEYS = [
    "spectrum_id",
    "source_file",
    "task",
    "scan",
    "ms_level",
    "library_membership",
    "spectrum_status",
    "peaks_json",
    "splash",
    "submit_user",
    "Compound_Name",
    "Ion_Source",
    "Compound_Source",
    "Instrument",
    "PI",
    "Data_Collector",
    "Adduct",
    "Scan",
    "Precursor_MZ",
    "ExactMass",
    "Charge",
    "CAS_Number",
    "Pubmed_ID",
    "Smiles",
    "INCHI",
    "INCHI_AUX",
    "Library_Class",
    "SpectrumID",
    "Ion_Mode",
    "create_time",
    "task_id",
    "user_id",
    "InChIKey_smiles",
    "InChIKey_inchi",
    "Formula_smiles",
    "Formula_inchi",
    "url",
]


class FSInputDataGateway(InputDataGateway):
    def __init__(self, fs: Optional[FileSystemBase] = None):
        self.fs = fs

    def download_gnps(self, uri: str, output_path: str):
        if self.fs is None:
            self.fs = get_fs(output_path)
        output_path = DRPath(output_path)
        self._download_and_serialize(uri, output_path)

    def _download_and_serialize(self, uri: str, output_path: DRPath):
        """solution is from https://stackoverflow.com/a/16696317/15485553"""
        logger.info(f"Loading data from {uri}... This might take a while.")
        chunk_size = 10 * 1024 * 1024
        try:
            with requests.get(uri, stream=True) as r:
                r.raise_for_status()
                with self.fs.open(output_path, "wb") as f:
                    file_size = 0
                    for chunk in r.iter_content(
                        chunk_size=chunk_size
                    ):  # 10 MBs of chunks
                        f.write(chunk)
                        file_size += chunk_size
        except requests.ConnectionError:
            self._resume_download(file_size, uri, str(output_path))
        except requests.Timeout:
            self._resume_download(file_size, uri, str(output_path))
        except requests.HTTPError:
            self._resume_download(file_size, uri, str(output_path))
        except requests.TooManyRedirects:
            self._resume_download(file_size, uri, str(output_path))

    def _resume_download(self, file_size: int, uri: str, file_path: str):
        """solution is from https://stackoverflow.com/questions/22894211/how-to-resume-file-download-in-python """
        resume_header = {"Range": f"bytes={file_size}-"}
        with requests.get(uri, stream=True, headers=resume_header) as r:
            r.raise_for_status()
            with self.fs.open(file_path, "ab") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)

    def load_spectrum(self, path: str) -> SpectrumInputData:
        if self.fs is None:
            self.fs = get_fs(path)

        with self.fs.open(DRPath(path), "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            results = [{k: item[k] for k in KEYS} for item in items]
        return results

    def load_spectrum_ids(
        self, path: str, spectrum_ids: List[str]
    ) -> SpectrumInputData:
        spectrum_ids = set(spectrum_ids)
        if self.fs is None:
            self.fs = get_fs(path)

        with self.fs.open(DRPath(path), "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            results = [
                {k: item[k] for k in KEYS}
                for item in items
                if item["SpectrumID"] in spectrum_ids
            ]
        return results

    # TODO: docstring and test
    def chunk_gnps(self, gnps_path: str, chunk_size: int) -> List[str]:
        if self.fs is None:
            self.fs = get_fs(gnps_path)

        chunks_output_dir = DRPath(gnps_path).parent / "chunks"

        with self.fs.open(DRPath(gnps_path), "rb") as f:
            chunk = []
            chunk_ix = 0
            chunk_paths = []

            items = ijson.items(f, "item", multiple_values=True)
            for item in items:
                chunk.append({k: item[k] for k in KEYS})

                if len(chunk) == chunk_size:
                    chunk_path = chunks_output_dir / f"chunk_{chunk_ix}.json"
                    chunk_paths.append(chunk_path)
                    with self.fs.open(chunk_path, "wb") as f:
                        f.write(json.dumps(chunk).encode("UTF-8"))
                        chunk = []
                        chunk_ix += 1

            if chunk:
                chunk_path = chunks_output_dir / f"chunk_{chunk_ix}.json"
                chunk_paths.append(chunk_path)
                with self.fs.open(chunk_path, "wb") as f:
                    f.write(json.dumps(chunk).encode("UTF-8"))

        return chunk_paths

    def get_spectrum_ids(self, path: str) -> List[str]:
        if self.fs is None:
            self.fs = get_fs(path)

        with self.fs.open(DRPath(path), "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            ids = [item["SpectrumID"] for item in items]

        return ids

    # TODO: docstring and test
    def serialize_to_file(self, path: str, obj: Any) -> bool:
        if self.fs is None:
            self.fs = get_fs(path)

        with self.fs.open(path, "wb") as f:
            pickle.dump(obj, f)

        return True
