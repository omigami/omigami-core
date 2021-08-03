from dataclasses import dataclass
from typing import Dict, Set

from ms2deepscore.models import load_model as ms2deepscore_load_model
from omigami.config import IonModes
from omigami.gateways.redis_spectrum_data_gateway import (
    REDIS_DB,
)
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.helper_classes.embedding_maker import EmbeddingMaker
from omigami.ms2deepscore.helper_classes.ms2deepscore_embedding import (
    MS2DeepScoreEmbedding,
)
from omigami.utils import merge_prefect_task_configs
from prefect import Task


@dataclass
class MakeEmbeddingsParameters:
    ion_mode: IonModes


class MakeEmbeddings(Task):
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        parameters: MakeEmbeddingsParameters,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._embedding_maker = EmbeddingMaker()
        self._ion_mode = parameters.ion_mode

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        model_path: str = None,
        model_registry: Dict[str, str] = None,
        spectrum_ids: Set[str] = None,
    ) -> Set[str]:
        self.logger.info(f"Creating {len(spectrum_ids)} embeddings.")
        binned_spectra = self._spectrum_dgw.read_binned_spectra(
            self._ion_mode, spectrum_ids
        )
        self.logger.info(
            f"Loaded {len(binned_spectra)} binned spectra from the database."
        )

        embeddings = []
        siamese_model = ms2deepscore_load_model(model_path)
        model = MS2DeepScoreEmbedding(siamese_model)
        for i, binned_spectrum in enumerate(binned_spectra):
            embeddings.append(
                self._embedding_maker.make_embedding(
                    model,
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
            embeddings, model_registry["run_id"], self.logger
        )
        return spectrum_ids
