from typing import List

import mlflow
from mlflow.entities import Run
from prefect import Task

from omigami.spectra_matching.spec2vec.predictor import Spec2VecPredictor
from omigami.spectra_matching.storage import DataGateway
from omigami.spectra_matching.storage.model_registry import (
    ModelRegistryDataGateway,
)
from omigami.utils import merge_prefect_task_configs


class ChunkDocumentPaths(Task):
    def __init__(
        self,
        documents_directory: str,
        fs_dgw: DataGateway,
        chunk_size: int,
        **kwargs,
    ):
        self._fs_gtw = fs_dgw
        self._documents_directory = documents_directory
        self._chunk_size = chunk_size

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self) -> List[List[str]]:
        """Lists all paths to document pickle files and chunks them. Returns the chunked
        document paths."""
        self.logger.info(
            f"Reading document paths from director {self._documents_directory}."
        )
        document_paths = self._fs_gtw.list_files(self._documents_directory)
        chunked_paths = [
            document_paths[i * self._chunk_size : (i + 1) * self._chunk_size]
            for i in range(len(document_paths) // self._chunk_size + 1)
        ]
        self.logger.info(
            f"Created {len(chunked_paths)} chunks of size {self._chunk_size}."
        )
        return chunked_paths


# we could probably unify this task for ms2ds and spec2vec but it would require
# some changes on the way we are saving models in spec2vec's RegisterModel task.
class LoadSpec2VecModel(Task):
    def __init__(
        self,
        model_run_id: str,
        fs_dgw: DataGateway,
        model_registry_dgw: ModelRegistryDataGateway,
        **kwargs,
    ):
        self._model_run_id = model_run_id
        self._fs_gtw = fs_dgw
        self._model_registry_dgw = model_registry_dgw

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self) -> Spec2VecPredictor:
        """Loads a trained spec2vec model given an mlflow_run_id"""
        self.logger.info(f"Loading Spec2Vec model with run_id {self._model_run_id}.")
        run: Run = mlflow.get_run(self._model_run_id)
        model_path = f"{run.info.artifact_uri}/model/python_model.pkl"
        self.logger.info(f"Loading model from path: {model_path}")
        model = self._fs_gtw.read_from_file(model_path)
        return model
