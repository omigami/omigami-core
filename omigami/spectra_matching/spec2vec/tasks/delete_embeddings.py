from prefect import Task

from omigami.config import IonModes
from omigami.spectra_matching.storage import (
    RedisSpectrumDataGateway,
)
from omigami.utils import merge_prefect_task_configs


class DeleteEmbeddings(Task):
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        ion_mode: IonModes,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._ion_mode = ion_mode

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self):
        """
        Deletes the embeddings for the previous model of the same ion mode so that
        new embeddings can be created.

        Returns
        -------
        Set of spectrum_ids

        """
        self.logger.info(
            f"Deleting embeddings for spec2vec model of {self._ion_mode} ion mode"
        )
        self._spectrum_dgw.delete_embeddings(self._ion_mode)
