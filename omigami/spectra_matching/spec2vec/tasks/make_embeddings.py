from dataclasses import dataclass
from typing import Union, Set

from gensim.models import Word2Vec
from prefect import Task

from omigami.common.progress_logger import (
    TaskProgressLogger,
)
from omigami.config import IonModes
from omigami.spectra_matching.spec2vec.helper_classes.embedding_maker import (
    EmbeddingMaker,
)
from omigami.spectra_matching.storage import (
    RedisSpectrumDataGateway,
    FSDataGateway,
    REDIS_DB,
)
from omigami.utils import merge_prefect_task_configs


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
        fs_gtw: FSDataGateway,
        parameters: MakeEmbeddingsParameters,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._fs_gtw = fs_gtw
        self._embedding_maker = EmbeddingMaker(n_decimals=parameters.n_decimals)
        self._ion_mode = parameters.ion_mode
        self._intensity_weighting_power = parameters.intensity_weighting_power
        self._allowed_missing_percentage = parameters.allowed_missing_percentage

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        model: Word2Vec = None,
        model_run_id: str = None,
        document_path: str = None,
    ) -> Set[str]:
        """
        Prefect task to create embeddings with Word2Vec model trained on spectrum
        documents. Then they are saved to REDIS DB. Each embedding is a vector
        calculated by
            - spectrum document
            - `intensity_weighting_power`
            - `allowed_missing_percentage`
            - `n_decimals_for_documents`

        Parameters
        ----------
        model: Word2Vec
            Model trained on spectrum documents
        model_run_id:
            Registered model's `run_id`
        document_path: str
            Directory which documents are saved

        Returns
        -------
        Set of spectrum_ids

        """
        documents = self._fs_gtw.read_from_file(document_path)

        self.logger.info(f"Loaded {len(documents)} documents from filesystem.")

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
        self.logger.debug(f"Using Redis DB {REDIS_DB} and model id {model_run_id}.")
        self._spectrum_dgw.write_embeddings(embeddings, self._ion_mode, self.logger)
        return set(doc.get("spectrum_id") for doc in documents)
