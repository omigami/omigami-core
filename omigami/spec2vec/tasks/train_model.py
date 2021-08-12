from typing import List, Set

import gensim
import prefect
from attr import dataclass
from prefect import Task
from spec2vec.model_building import (
    set_spec2vec_defaults,
    learning_rates_to_gensim_style,
)

from omigami.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
)
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.helper_classes.train_logger import (
    CustomTrainingProgressLogger,
)
from omigami.utils import merge_prefect_task_configs


@dataclass
class TrainModelParameters:
    """
    iterations:
        Number of training iterations
    window:
        Window size for context around the word
    """

    epochs: int = 25
    window: int = 500


class TrainModel(Task):
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        training_parameters: TrainModelParameters,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._epochs = training_parameters.epochs
        self._window = training_parameters.window

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, documents_directory: List[str]):

        documents = self._load_all_document_files(documents_directory)

        for doc_dir in documents_directory:
            documents = documents + self._spectrum_dgw.read_documents(doc_dir)

        self.logger.info(
            f"Connecting to the data. {len(documents)} documents will be used on training."
        )
        self.logger.info("Started training the Word2Vec model.")
        callbacks, settings = self._create_spec2vec_settings(self._window, self._epochs)
        model = gensim.models.Word2Vec(
            sentences=documents, callbacks=callbacks, **settings
        )
        self.logger.info(f"Finished training the model.")

        return model

    def _create_spec2vec_settings(self, window: int, epochs: int):
        settings = set_spec2vec_defaults(window=window)

        # Convert spec2vec style arguments to gensim style arguments
        settings = learning_rates_to_gensim_style(num_of_epochs=epochs, **settings)

        # Set callbacks
        callbacks = []
        training_progress_logger = CustomTrainingProgressLogger(epochs, self.logger)
        callbacks.append(training_progress_logger)

        return callbacks, settings

    def _load_all_document_files(
        self, documents_directory: List[str]
    ) -> List[SpectrumDocumentData]:

        documents = []
        for doc_dir in documents_directory:
            docs = self._spectrum_dgw.read_documents(doc_dir)
            documents = documents + docs

        return documents
