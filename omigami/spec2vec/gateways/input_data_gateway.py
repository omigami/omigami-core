import json
import logging
import pickle
import sys
from typing import List, Optional, Any

import ijson
import requests
from drfs import DRPath
from drfs.filesystems import get_fs
from drfs.filesystems.base import FileSystemBase

from omigami.config import IonModes
from omigami.gateways.data_gateway import InputDataGateway
from omigami.spec2vec.entities.data_models import SpectrumInputData

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

    def download_ms2deep_model(self, uri: str, output_path: str):
        if self.fs is None:
            self.fs = get_fs(output_path)
        try:
            req = requests.get(url=uri)
            req.raise_for_status()
            logging.info(f"Response: {req.status_code} - {req.reason}")
            logging.info(f"Response: {req.content}")

            with self.fs.open(DRPath(output_path), mode="wb") as f:
                f.write(req.content)
        except Exception:
            logging.error("Unable to download ms2deepscore model")
            raise Exception

    def _download_and_serialize(self, uri: str, output_path: DRPath):
        """solution is from https://stackoverflow.com/a/16696317/15485553"""
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
        """solution is from https://stackoverflow.com/questions/22894211/how-to-resume-file-download-in-python"""
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

    def chunk_gnps(
        self,
        gnps_path: str,
        chunk_size: int,
        ion_mode: IonModes = "positive",
        logger: logging.Logger = None,
    ) -> List[str]:
        """
        The chunking works as following:
        1. Open a stream to the gnps_path json file
        2. Start looping through the spectra and appending each one to a list
        3. When the size of the list reaches `chunk_size`:
          a. save the list to a json identified by the chunk index
          b. empty the list to start looping again
          c. add the path to the chunk that was just saved to a list of paths
        4. Repeat the previous steps until all file has been read

        Parameters
        ----------
        gnps_path:
            Path to the gnps file
        chunk_size:
            Size in bytes of each chunk
        ion_mode:
            Ion mode selected for training (positive or negative)
        logger:
            Optional logger for progress

        Returns
        -------
        List of paths:
            A list of paths for the saved chunked files

        """

        if self.fs is None:
            self.fs = get_fs(gnps_path)

        chunks_output_dir = f"{str(DRPath(gnps_path).parent)}/chunks/{ion_mode}"

        with self.fs.open(DRPath(gnps_path), "rb") as gnps_file:
            chunk = []
            chunk_ix = 0
            chunk_paths = []
            chunk_bytes = 0

            items = ijson.items(gnps_file, "item", multiple_values=True)
            for item in items:
                spectrum = {k: item[k] for k in KEYS}
                if spectrum["Ion_Mode"].lower() != ion_mode:
                    continue

                chunk.append(spectrum)
                chunk_bytes += sys.getsizeof(spectrum) + sys.getsizeof(
                    spectrum["peaks_json"]
                )

                if chunk_bytes >= chunk_size:
                    chunk_path = f"{chunks_output_dir}/chunk_{chunk_ix}.json"
                    chunk_paths.append(chunk_path)

                    with self.fs.open(chunk_path, "wb") as chunk_file:
                        chunk_file.write(json.dumps(chunk).encode("UTF-8"))
                        chunk = []
                        chunk_ix += 1
                        chunk_bytes = 0

                    if logger:
                        logger.info(f"Saved chunk to path {chunk_path}.")

            if chunk:
                chunk_path = f"{chunks_output_dir}/chunk_{chunk_ix}.json"
                chunk_paths.append(chunk_path)
                with self.fs.open(chunk_path, "wb") as chunk_file:
                    chunk_file.write(json.dumps(chunk).encode("UTF-8"))

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
        path = DRPath(path)
        if self.fs is None:
            self.fs = get_fs(path)

        with self.fs.open(path, "wb") as f:
            pickle.dump(obj, f)

        return True
