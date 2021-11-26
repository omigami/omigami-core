from dataclasses import dataclass
from typing import Dict, Set

from prefect import Task

from omigami.config import IonModes
from omigami.spectra_matching.ms2deepscore.gateways import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.gateways.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.ms2deepscore.similarity_score_calculator import (
    MS2DeepScoreSimilarityScoreCalculator,
    EmbeddingMaker,
)
from omigami.spectra_matching.storage import REDIS_DB
from omigami.utils import merge_prefect_task_configs


@dataclass
class MakeEmbeddingsParameters:
    ion_mode: IonModes


class MakeEmbeddings(Task):
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        fs_gtw: MS2DeepScoreFSDataGateway,
        parameters: MakeEmbeddingsParameters,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._fs_gtw = fs_gtw
        self._embedding_maker = EmbeddingMaker()
        self._ion_mode = parameters.ion_mode

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        train_model_output: Dict = None,
        model_registry: Dict[str, str] = None,
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
        model_registry: Dict[str, str]
            Dictionary containing registered `model_uri` and `run_id`
        spectrum_ids: Set[str]
            Set of spectrum_ids to make embedding from

        Returns
        -------
        spectrum_ids : Set[str]
            Set of spectrum_ids

        """
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
        similarity_score_calculator = MS2DeepScoreSimilarityScoreCalculator(
            siamese_model
        )

        for i, binned_spectrum in enumerate(binned_spectra):
            embeddings.append(
                self._embedding_maker.make_embedding(
                    similarity_score_calculator,
                    binned_spectrum,
                )
            )

        self.logger.info(
            f"Finished creating embeddings. Saving {len(embeddings)} embeddings to "
            f"database."
        )
        self.logger.debug(
            f"Using Redis DB {REDIS_DB} and model id {model_registry['run_id']}."
        )
        self._spectrum_dgw.write_embeddings(
            embeddings, self._ion_mode, model_registry["run_id"], self.logger
        )
        return spectrum_ids
