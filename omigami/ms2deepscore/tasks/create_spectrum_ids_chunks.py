from dataclasses import dataclass
from typing import List, Set

from omigami.utils import merge_prefect_task_configs
from prefect import Task


@dataclass
class ChunkingIDsParameters:
    spectrum_ids_chunk_size: int


class CreateSpectrumIDsChunks(Task):
    def __init__(
        self,
        chunking_parameters: ChunkingIDsParameters,
        **kwargs,
    ):
        self._chunk_size = chunking_parameters.spectrum_ids_chunk_size

        config = merge_prefect_task_configs(kwargs)

        super().__init__(**config)

    def run(self, spectrum_ids: Set[str] = None) -> List[List[str]]:
        spectrum_ids = list(spectrum_ids)
        chunks = [
            spectrum_ids[x : x + self._chunk_size]
            for x in range(0, len(spectrum_ids), self._chunk_size)
        ]

        self.logger.info(
            f"Split spectra into {len(chunks)} chunks of size {self._chunk_size}"
        )

        return chunks
