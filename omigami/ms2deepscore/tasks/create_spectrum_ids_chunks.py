import numpy as np
from dataclasses import dataclass
from typing import List
from prefect import Task
from omigami.gateways import RedisSpectrumDataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class ChunkingParameters:
    n_chunks: int


class CreateSpectrumIDsChunks(Task):
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        chunking_parameters: ChunkingParameters,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._n_chunks = chunking_parameters.n_chunks

        config = merge_prefect_task_configs(kwargs)

        super().__init__(**config)

    def run(self) -> List[List[str]]:
        spectrum_ids = self._spectrum_dgw.list_spectrum_ids()
        self.logger.info(f"There are {len(spectrum_ids)} in the database.")

        chunks = np.split(np.array(spectrum_ids), self._n_chunks)

        self.logger.info(
            f"Split spectra into {len(chunks)} chunks of size {self._n_chunks}"
        )

        return chunks
