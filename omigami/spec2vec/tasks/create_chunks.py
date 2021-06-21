from dataclasses import dataclass
from typing import List

from drfs import DRPath
from prefect import Task

from omigami.config import IonModes
from omigami.data_gateway import InputDataGateway
from omigami.utils import create_prefect_result_from_path, merge_prefect_task_configs


@dataclass
class ChunkingParameters:
    file_path: str
    chunk_size: int
    ion_mode: IonModes

    @property
    def chunk_paths_file(self) -> str:
        return (
            f"{str(DRPath(self.file_path).parent)}/"
            f"chunks/{self.ion_mode}/chunk_paths.pickle"
        )


class CreateChunks(Task):
    def __init__(
        self,
        input_dgw: InputDataGateway,
        chunking_parameters: ChunkingParameters,
        **kwargs,
    ):
        self._input_dgw = input_dgw
        self._chunk_size = chunking_parameters.chunk_size
        self._file_path = chunking_parameters.file_path
        self._ion_mode = chunking_parameters.ion_mode
        self._chunk_paths_file = chunking_parameters.chunk_paths_file

        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
            **create_prefect_result_from_path(chunking_parameters.chunk_paths_file),
            checkpoint=True,
        )

    def run(self, spectrum_ids: List[str] = None) -> List[str]:
        self.logger.info(f"Loading file {self._file_path} for chunking.")
        # maybe we should save which ids are on each chunk in the pickle file
        chunk_paths = self._input_dgw.chunk_gnps(
            self._file_path, self._chunk_size, self._ion_mode, self.logger
        )
        self.logger.info(
            f"Split spectra into {len(chunk_paths)} chunks of size"
            f"{self._chunk_size}"
        )

        self.logger.info(f"Saving pickle with file paths to {self._chunk_paths_file}")
        self._input_dgw.serialize_to_file(self._chunk_paths_file, chunk_paths)

        return chunk_paths