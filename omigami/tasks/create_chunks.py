from typing import List

from drfs import DRPath
from prefect import Task

from omigami.config import ION_MODES, IonModes
from omigami.data_gateway import InputDataGateway
from omigami.tasks.config import merge_configs


class CreateChunks(Task):
    def __init__(
        self,
        file_path: str,
        input_dgw: InputDataGateway,
        chunk_size: int,
        ion_mode: IonModes = "positive",
        **kwargs,
    ):
        self._input_dgw = input_dgw
        self._chunk_size = chunk_size
        self._file_path = file_path  # dataset-id/chunks -> dataset-id/ion-mode/chunks/
        self._ion_mode = ion_mode

        if ion_mode not in ION_MODES:
            raise ValueError("Ion mode can only be either 'positive' or 'negative'.")

        config = merge_configs(kwargs)

        super().__init__(**config, checkpoint=True)

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

        chunk_paths_file = f"{str(DRPath(self._file_path).parent)}/chunk_paths.pickle"
        self.logger.info(f"Saving pickle with file paths to {chunk_paths_file}")
        self._input_dgw.serialize_to_file(chunk_paths_file, chunk_paths)

        return chunk_paths
