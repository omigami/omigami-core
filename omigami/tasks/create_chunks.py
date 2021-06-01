from typing import List, Optional

from prefect import Task

from omigami.tasks.config import DEFAULT_CONFIG


class CreateChunks(Task):
    def __init__(
        self,
        chunk_size: Optional[int] = None,
    ):
        self._chunk_size = chunk_size

        super().__init__(**DEFAULT_CONFIG)

    def run(self, spectrum_ids: List[str] = None) -> List[str]:
        chunked_spectrum_data = [
            spectrum_ids[i : i + self._chunk_size]
            for i in range(0, len(spectrum_ids), self._chunk_size)
        ]
        self.logger.info(
            f"Split spectra into {len(chunked_spectrum_data)} chunks of size"
            f"{self._chunk_size}"
        )
        return chunked_spectrum_data
