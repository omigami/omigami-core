from typing import Union, List, Dict

from gensim.models import Word2Vec
from prefect import Task

from omigami.helper_classes.embedding_maker import EmbeddingMaker
from omigami.helper_classes.progress_logger import TaskProgressLogger
from omigami.tasks.config import merge_configs
from omigami.data_gateway import SpectrumDataGateway


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
        spectrum_ids: List[str] = None,
    ) -> List[str]:
        self.logger.info(f"Creating {len(spectrum_ids)} embeddings.")
        documents = self._spectrum_dgw.read_documents(spectrum_ids)
        self.logger.info(f"Loaded {len(documents)} embeddings from the database.")

        embeddings = []
        progress_logger = TaskProgressLogger(
            self.logger, len(documents), 25, "Make Embeddings task progress"
        )
        for i, document in enumerate(documents):
            embedding = self._embedding_maker.make_embedding(
                model,
                document,
                self._intensity_weighting_power,
                self._allowed_missing_percentage,
            )
            embeddings.append(embedding)
            progress_logger.log(i)

        self.logger.info("Finished creating embeddings. Saving embeddings to database.")
        self._spectrum_dgw.write_embeddings(embeddings, model_registry["run_id"])
        return spectrum_ids
