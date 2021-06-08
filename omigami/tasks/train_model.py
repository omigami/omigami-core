from typing import List

import gensim
import prefect
from attr import dataclass
from prefect import Task
from spec2vec.model_building import (
    set_spec2vec_defaults,
    learning_rates_to_gensim_style,
)
from spec2vec.utils import TrainingProgressLogger

from omigami.data_gateway import SpectrumDataGateway
from omigami.helper_classes.train_logger import CustomTrainingProgressLogger
from omigami.tasks.config import merge_configs


@dataclass
class TrainModelParameters:
    """
    iterations:
        Number of training iterations
    window:
        Window size for context around the word
    """

    spectrum_dgw: SpectrumDataGateway
    epochs: int = 25
    window: int = 500

    @property
    def kwargs(self):
        return {
            "spectrum_dgw": self.spectrum_dgw,
            "epochs": self.epochs,
            "window": self.window,
        }


class TrainModel(Task):
    def __init__(
        self,
        spectrum_dgw: SpectrumDataGateway,
        epochs: int = 25,
        window: int = 500,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._epochs = epochs
        self._window = window

        config = merge_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, spectrum_ids_chunks: List[List[str]] = None):
        flatten_ids = [item for elem in spectrum_ids_chunks for item in elem]
        self.logger.info(
            f"Connecting to the data. {len(flatten_ids)} documents will be used on training."
        )
        documents = self._spectrum_dgw.read_documents_iter(flatten_ids)

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
