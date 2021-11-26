from typing import List

import gensim
import prefect
from attr import dataclass
from gensim.models import Word2Vec
from prefect import Task
from spec2vec.model_building import (
    set_spec2vec_defaults,
    learning_rates_to_gensim_style,
)

from omigami.spectra_matching.spec2vec.helper_classes.train_logger import (
    CustomTrainingProgressLogger,
)
from omigami.spectra_matching.spec2vec.storage.fs_document_iterator import (
    FileSystemDocumentIterator,
)
from omigami.spectra_matching.storage import FSDataGateway
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
        data_dgw: FSDataGateway,
        training_parameters: TrainModelParameters,
        **kwargs,
    ):
        self._data_dgw = data_dgw
        self._epochs = training_parameters.epochs
        self._window = training_parameters.window

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, document_paths: List[str] = None) -> Word2Vec:
        """
        Prefect task to train a Word2Vec model with the spectrum documents.

        Parameters
        ----------
        document_paths: List[str]
            Path to read spectrum documents from

        Returns
        -------
        model: Word2Vec

        """

        self.logger.info(f"Loading examples from {document_paths}")

        documents = FileSystemDocumentIterator(
            fs_dgw=self._data_dgw, document_paths=document_paths
        )

        self.logger.info("Started training the Word2Vec model.")
        callbacks, settings = self._create_spec2vec_settings(self._window, self._epochs)
        model = gensim.models.Word2Vec(
            sentences=documents, callbacks=callbacks, **settings
        )
        self.logger.info(f"Trained model on {len(documents)} examples.")
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
