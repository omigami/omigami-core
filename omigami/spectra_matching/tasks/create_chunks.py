import json
import sys
from dataclasses import dataclass
from typing import List

import ijson
from drfs import DRPath
from drfs.filesystems import get_fs
from prefect import Task

from omigami.config import IonModes
from omigami.spectra_matching.storage import DataGateway, KEYS
from omigami.utils import create_prefect_result_from_path, merge_prefect_task_configs


@dataclass
class ChunkingParameters:
    input_file: str
    output_directory: str
    chunk_size: int
    ion_mode: IonModes

    @property
    def checkpoint_file(self) -> str:
        return f"{self.output_directory}/chunk_paths.pickle"


class CreateChunks(Task):
    def __init__(
        self,
        data_gtw: DataGateway,
        chunking_parameters: ChunkingParameters,
        **kwargs,
    ):
        self._data_gtw = data_gtw
        self._chunk_size = chunking_parameters.chunk_size
        self._input_file = chunking_parameters.input_file
        self._output_directory = chunking_parameters.output_directory
        self._ion_mode = chunking_parameters.ion_mode
        self._checkpoint_file = chunking_parameters.checkpoint_file

        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
            **create_prefect_result_from_path(chunking_parameters.checkpoint_file),
            checkpoint=True,
        )

    def run(self, spectrum_ids: List[str] = None) -> List[str]:
        """
        Prefect task to split GNPS data into chunks. First, GNPS data file is read from
        filesystem. Then the file is split, and smaller files are created in `chunk_size`.
        Finally, chunked files are saved to filesystem. This is necessary to
        parallelize task orchestration.

        Parameters
        ----------
        spectrum_ids: List[str]

        Returns
        -------
        chunked_paths: List[str]
            Paths of chunked files

        """
        self.logger.info(f"Loading file {self._input_file} for chunking.")
        chunk_paths = self.chunk_gnps(self._input_file)
        self.logger.info(
            f"Split spectra into {len(chunk_paths)} chunks of size"
            f"{self._chunk_size}"
        )

        self.logger.info(f"Saving pickle with file paths to {self._checkpoint_file}")
        self._data_gtw.serialize_to_file(self._checkpoint_file, chunk_paths)

        return chunk_paths

    def chunk_gnps(self, gnps_path: str) -> List[str]:
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

        Returns
        -------
        List of paths:
            A list of paths for the saved chunked files

        """

        fs = get_fs(gnps_path)

        with fs.open(DRPath(gnps_path), "rb") as gnps_file:
            chunk = []
            chunk_ix = 0
            chunk_paths = []
            chunk_bytes = 0

            items = ijson.items(gnps_file, "item", multiple_values=True)
            for item in items:
                spectrum = {k: item[k] for k in KEYS}
                if spectrum["Ion_Mode"].lower() != self._ion_mode:
                    continue

                chunk.append(spectrum)
                chunk_bytes += sys.getsizeof(spectrum) + sys.getsizeof(
                    spectrum["peaks_json"]
                )

                if chunk_bytes >= self._chunk_size:
                    chunk_path = f"{self._output_directory}/chunk_{chunk_ix}.json"
                    chunk_paths.append(chunk_path)

                    with fs.open(chunk_path, "wb") as chunk_file:
                        chunk_file.write(json.dumps(chunk).encode("UTF-8"))
                        chunk = []
                        chunk_ix += 1
                        chunk_bytes = 0

                    self.logger.info(f"Saved chunk to path {chunk_path}.")

            if chunk:
                chunk_path = f"{self._output_directory}/chunk_{chunk_ix}.json"
                chunk_paths.append(chunk_path)
                with fs.open(chunk_path, "wb") as chunk_file:
                    chunk_file.write(json.dumps(chunk).encode("UTF-8"))

        return chunk_paths
