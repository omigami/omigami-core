from typing import Union, List, Dict, Set

from gensim.models import Word2Vec
from prefect import Task

from omigami.data_gateway import SpectrumDataGateway
from omigami.gateways.redis_spectrum_gateway import REDIS_DB
from omigami.helper_classes.embedding_maker import EmbeddingMaker
from omigami.helper_classes.progress_logger import TaskProgressLogger
from omigami.tasks.config import merge_configs


class MakeEmbeddings(Task):
    def __init__(
        self,
        spectrum_dgw: SpectrumDataGateway,
        n_decimals: int,
        intensity_weighting_power: Union[float, int] = 0.5,
        allowed_missing_percentage: Union[float, int] = 5.0,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._embedding_maker = EmbeddingMaker(n_decimals=n_decimals)
        self._intensity_weighting_power = intensity_weighting_power
        self._allowed_missing_percentage = allowed_missing_percentage

        config = merge_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        model: Word2Vec = None,
        model_registry: Dict[str, str] = None,
        spectrum_ids: Set[str] = None,
    ) -> List[str]:
        self.logger.info(f"Creating {len(spectrum_ids)} embeddings.")
        documents = self._spectrum_dgw.read_documents(spectrum_ids)
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
        self._spectrum_dgw.write_embeddings(
            embeddings, model_registry["run_id"], self.logger
        )
        return spectrum_ids
