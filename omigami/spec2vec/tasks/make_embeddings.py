from dataclasses import dataclass
from typing import Union, Dict, Set

from gensim.models import Word2Vec
from omigami.config import IonModes
from omigami.gateways.redis_spectrum_data_gateway import (
    REDIS_DB,
    RedisSpectrumDataGateway,
)
from omigami.spec2vec.helper_classes.embedding_maker import EmbeddingMaker
from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger
from omigami.utils import merge_prefect_task_configs
from prefect import Task


@dataclass
class MakeEmbeddingsParameters:
    ion_mode: IonModes
    n_decimals: int
    intensity_weighting_power: Union[float, int] = 0.5
    allowed_missing_percentage: Union[float, int] = 5.0


class MakeEmbeddings(Task):
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        parameters: MakeEmbeddingsParameters,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._embedding_maker = EmbeddingMaker(n_decimals=parameters.n_decimals)
        self._ion_mode = parameters.ion_mode
        self._intensity_weighting_power = parameters.intensity_weighting_power
        self._allowed_missing_percentage = parameters.allowed_missing_percentage

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        model: Word2Vec = None,
        model_registry: Dict[str, str] = None,
        spectrum_ids: Set[str] = None,
    ) -> Set[str]:
        self.logger.info(f"Creating {len(spectrum_ids)} embeddings.")
        documents = self._spectrum_dgw.read_documents(spectrum_ids)
        self.logger.info(f"Loaded {len(documents)} documents from the database.")

        embeddings = []
        progress_logger = TaskProgressLogger(
            self.logger, len(documents), 25, "Make Embeddings task progress"
        )

        i = 0
        for document in documents:
            embeddings.append(
                self._embedding_maker.make_embedding(
                    model,
                    document,
                    self._intensity_weighting_power,
                    self._allowed_missing_percentage,
                )
            )
            progress_logger.log(i)
            i += 1

        self.logger.info(
            f"Finished creating embeddings. Saving {len(embeddings)} embeddings to database."
        )
        self.logger.debug(
            f"Using Redis DB {REDIS_DB} and model id {model_registry['run_id']}."
        )
        self._spectrum_dgw.write_embeddings(
            embeddings, self._ion_mode, model_registry["run_id"], self.logger
        )
        return spectrum_ids
