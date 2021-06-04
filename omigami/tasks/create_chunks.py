from typing import List, Optional

from prefect import Task

from omigami.data_gateway import SpectrumDataGateway
from omigami.tasks.config import merge_configs


class CreateChunks(Task):
    def __init__(
        self,
        file_path: str,
        output_dir: str,
        input_dgw: SpectrumDataGateway,
        chunk_size: Optional[int] = None,
        **kwargs,
    ):
        self._input_dgw = input_dgw
        self._chunk_size = chunk_size
        self._file_path = file_path
        self._output_dir = output_dir

        config = merge_configs(kwargs)

        super().__init__(**config, checkpoint=True)

    def run(self, spectrum_ids: List[str] = None) -> List[str]:
        chunk_paths = self._input_dgw.chunk_gnps(self._file_path, self._chunk_size)
        self.logger.info(
            f"Split spectra into {len(chunk_paths)} chunks of size"
            f"{self._chunk_size}"
        )

        chunk_paths_file = f"{self._output_dir}/chunk_paths.pickle"
        self.logger.info(f"Saving pickle with file paths to {chunk_paths_file}")
        self._input_dgw.serialize_to_file(chunk_paths_file, chunk_paths)

        return chunk_paths
