from typing import List, Any, Dict

import gensim
import prefect
from attr import dataclass
from gensim.models import Word2Vec
from prefect import Task
from spec2vec.model_building import (
    set_spec2vec_defaults,
    learning_rates_to_gensim_style,
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

    model_directory: str
    epochs: int = 25
    window: int = 500

    @property
    def model_tmp_path(self) -> str:
        return self.model_directory + "/tmp/{flow_run_id}/word2vec.pickle"


class TrainModel(Task):
    def __init__(
        self,
        fs_dgw: FSDataGateway,
        training_parameters: TrainModelParameters,
        **kwargs,
    ):
        self._fs_dgw = fs_dgw
        self._model_tmp_path = training_parameters.model_tmp_path
        self._epochs = training_parameters.epochs
        self._window = training_parameters.window

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, document_paths: List[str] = None) -> str:
        """
        Prefect task to train a Word2Vec model with the spectrum documents.

        Parameters
        ----------
        document_paths: List[str]
            Path to read spectrum documents from

        Returns
        -------
        Path to the saved model.

        """

        self.logger.info(f"Loading documents from {document_paths}")

        documents = FileSystemDocumentIterator(
            fs_dgw=self._fs_dgw, document_paths=document_paths
        )

        self.logger.info(
            "Started training the Word2Vec model on {len(documents)} documents."
        )
        settings = self._create_spec2vec_settings()
        model = gensim.models.Word2Vec(sentences=documents, **settings)
        output_path = self._model_tmp_path.format(
            flow_run_id=prefect.context.get("flow_run_id", "local")
        )
        self.logger.info(f"Finished training the model. Saving model to {output_path}")
        self._fs_dgw.serialize_to_file(output_path, model)

        return output_path

    def _create_spec2vec_settings(self) -> Dict[str, Any]:
        settings = set_spec2vec_defaults(window=self._window)

        # Convert spec2vec style arguments to gensim style arguments
        settings = learning_rates_to_gensim_style(
            num_of_epochs=self._epochs, **settings
        )

        return settings
