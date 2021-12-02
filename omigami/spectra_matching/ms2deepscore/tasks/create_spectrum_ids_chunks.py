from dataclasses import dataclass
from typing import List, Set

from prefect import Task

from omigami.spectra_matching.storage import RedisSpectrumDataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class ChunkingIDsParameters:
    spectrum_ids_chunk_size: int


class CreateSpectrumIDsChunks(Task):
    def __init__(
        self,
        chunk_size: int,
        spectrum_dgw: RedisSpectrumDataGateway,
        **kwargs,
    ):
        self._chunk_size = chunk_size
        self._spectrum_dgw = spectrum_dgw

        config = merge_prefect_task_configs(kwargs)

        super().__init__(**config)

    def run(self, spectrum_ids: Set[str] = None) -> List[List[str]]:
        """
        Prefect task to split spectrum_ids into chunks. This is necessary to parallelize
        task orchestration.

        Parameters
        ----------
        spectrum_ids: Set[str]
            Set of spectrum_ids to split

        Returns
        -------
        chunks: List[List[str]]
            Chunked spectrum_ids, as a list of list containing spectrum_ids

        """
        if spectrum_ids is not None:
            spectrum_ids = list(spectrum_ids)
        else:
            spectrum_ids = self._spectrum_dgw.list_spectrum_ids()

        self.logger.info(f"Creating chunkfs for {len(spectrum_ids)} spectra.")
        chunks = [
            spectrum_ids[x : x + self._chunk_size]
            for x in range(0, len(spectrum_ids), self._chunk_size)
        ]

        self.logger.info(
            f"Split spectra into {len(chunks)} chunks of size {self._chunk_size}"
        )

        return chunks
