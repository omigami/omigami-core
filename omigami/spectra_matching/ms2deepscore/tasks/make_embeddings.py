from typing import Dict, Set

from prefect import Task

from omigami.config import IonModes
from omigami.spectra_matching.ms2deepscore.embedding import EmbeddingMaker
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.storage import REDIS_DB
from omigami.utils import merge_prefect_task_configs


class MakeEmbeddings(Task):
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        fs_gtw: MS2DeepScoreFSDataGateway,
        ion_mode: IonModes,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._fs_gtw = fs_gtw
        self._embedding_maker = EmbeddingMaker()
        self._ion_mode = ion_mode

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        train_model_output: Dict = None,
        run_id: str = None,
        spectrum_ids: Set[str] = None,
    ) -> Set[str]:
        """
        Prefect task to create embeddings from SiameseModel. First, binned spectra are
        read from DB for given spectrum_ids. Then, for each binned spectra, embeddings
        are created using SiameseModel and saved to REDIS DB. Resulting object is
        an `Embedding` object holding an embedding vector.

        Parameters
        ----------
        train_model_output: Dict[str, str]
            Dictionary containing `ms2deepscore_model_path` and `validation_loss`
        run_id:
            Registered model `run_id`
        spectrum_ids: Set[str]
            Set of spectrum_ids to make embedding from

        Returns
        -------
        spectrum_ids : Set[str]
            Set of spectrum_ids

        """
        self.logger.info(
            f"Deleting embeddings for spec2vec model of {self._ion_mode} ion mode"
        )
        self._spectrum_dgw.delete_embeddings(self._ion_mode)

        model_path = train_model_output["ms2deepscore_model_path"]
        self.logger.info(f"Creating {len(spectrum_ids)} embeddings.")
        binned_spectra = self._spectrum_dgw.read_binned_spectra(
            self._ion_mode, spectrum_ids
        )
        self.logger.info(
            f"Loaded {len(binned_spectra)} binned spectra from the database."
        )

        embeddings = []
        siamese_model = self._fs_gtw.load_model(model_path)

        for i, binned_spectrum in enumerate(binned_spectra):
            embeddings.append(
                self._embedding_maker.make_embedding(siamese_model, binned_spectrum)
            )

        self.logger.info(
            f"Finished creating embeddings. Saving {len(embeddings)} embeddings to "
            f"database."
        )
        self.logger.debug(f"Using Redis DB {REDIS_DB} and model id {run_id}.")
        self._spectrum_dgw.write_embeddings(embeddings, self._ion_mode, self.logger)
        return spectrum_ids
