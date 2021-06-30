from typing import List, Set

import gensim
import prefect
from attr import dataclass
from prefect import Task
from spec2vec.model_building import (
    set_spec2vec_defaults,
    learning_rates_to_gensim_style,
)

from omigami.gateways.data_gateway import SpectrumDataGateway
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
        spectrum_dgw: SpectrumDataGateway,
        training_parameters: TrainModelParameters,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._epochs = training_parameters.epochs
        self._window = training_parameters.window

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, spectrum_ids_chunks: List[Set[str]] = None):
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
