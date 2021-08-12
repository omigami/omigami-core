from dataclasses import dataclass
from typing import Union, Dict, Set, List

from gensim.models import Word2Vec
from omigami.config import IonModes
from omigami.gateways.redis_spectrum_data_gateway import (
    REDIS_DB,
    RedisSpectrumDataGateway,
)
from omigami.spec2vec.gateways import Spec2VecFSDocumentDataGateway
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
        redis_spectrum_dgw: RedisSpectrumDataGateway,
        fs_document_dgw: Spec2VecFSDocumentDataGateway,
        parameters: MakeEmbeddingsParameters,
        **kwargs,
    ):
        self._redis_spectrum_dgw = redis_spectrum_dgw
        self._fs_document_dgw = fs_document_dgw
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
        processed_document_paths: str = None,
    ) -> Set[str]:

        documents = self._fs_document_dgw.read_from_file(processed_document_paths)

        self.logger.info(f"Creating {len(documents)} embeddings.")
        self.logger.info(f"Loaded {len(documents)} documents from the database.")

        embeddings = []
        progress_logger = TaskProgressLogger(
            self.logger, len(documents), 25, "Make Embeddings task progress"
        )

        for i, document in enumerate(documents):
            embeddings.append(
                self._embedding_maker.make_embedding(
                    model,
                    document,
                    self._intensity_weighting_power,
                    self._allowed_missing_percentage,
                )
            )
            progress_logger.log(i)

        self.logger.info(
            f"Finished creating embeddings. Saving {len(embeddings)} embeddings to database."
        )
        self.logger.debug(
            f"Using Redis DB {REDIS_DB} and model id {model_registry['run_id']}."
        )
        self._redis_spectrum_dgw.write_embeddings(
            embeddings, self._ion_mode, model_registry["run_id"], self.logger
        )
        return set(doc.metadata["spectrum_id"] for doc in documents)
