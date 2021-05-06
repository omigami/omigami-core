from typing import List, Optional

from prefect import Task

from spec2vec_mlops.entities.data_models import SpectrumInputData
from spec2vec_mlops.tasks.config import DEFAULT_CONFIG


class CreateChunks(Task):
    def __init__(
        self,
        chunk_size: Optional[int] = None,
    ):
        self._chunk_size = chunk_size

        super().__init__(**DEFAULT_CONFIG)

    def run(self, spectrum_data: SpectrumInputData = None) -> List[SpectrumInputData]:
        chunked_spectrum_data = [
            spectrum_data[i : i + self._chunk_size]
            for i in range(0, len(spectrum_data), self._chunk_size)
        ]
        return chunked_spectrum_data
