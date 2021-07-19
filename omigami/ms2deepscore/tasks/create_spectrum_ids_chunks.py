from dataclasses import dataclass
from typing import List
from prefect import Task
from omigami.gateways import RedisSpectrumDataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class ChunkingParameters:
    chunk_spectrum_ids_size: int


class CreateSpectrumIDsChunks(Task):
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        chunking_parameters: ChunkingParameters,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._chunk_size = chunking_parameters.chunk_spectrum_ids_size

        config = merge_prefect_task_configs(kwargs)

        super().__init__(**config)

    def run(self) -> List[List[str]]:
        spectrum_ids = self._spectrum_dgw.list_spectrum_ids()
        self.logger.info(f"There are {len(spectrum_ids)} in the database.")

        chunks = [
            spectrum_ids[x : x + self._chunk_size]
            for x in range(0, len(spectrum_ids), self._chunk_size)
        ]

        self.logger.info(
            f"Split spectra into {len(chunks)} chunks of size" f"{self._chunk_size}"
        )

        return chunks
