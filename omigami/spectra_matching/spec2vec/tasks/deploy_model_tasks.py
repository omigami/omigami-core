from typing import List

import mlflow
from gensim.models import Word2Vec
from mlflow.entities import Run
from prefect import Task

from omigami.config import MLFLOW_SERVER
from omigami.spectra_matching.storage import DataGateway
from omigami.utils import merge_prefect_task_configs


class ListDocumentPaths(Task):
    def __init__(
        self,
        documents_directory: str,
        fs_dgw: DataGateway,
        **kwargs,
    ):
        self._fs_gtw = fs_dgw
        self._documents_directory = documents_directory

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self) -> List[str]:
        """Lists all paths to document pickle files."""
        self.logger.info(
            f"Reading document paths from directory {self._documents_directory}."
        )
        document_paths = self._fs_gtw.list_files(self._documents_directory)
        self.logger.info(f"Found {len(document_paths)} on directory.")
        return document_paths


# we could probably unify this task for ms2ds and spec2vec but it would require
# some changes on the way we are saving models in spec2vec's RegisterModel task.
class LoadSpec2VecModel(Task):
    def __init__(
        self,
        fs_dgw: DataGateway,
        **kwargs,
    ):
        self._fs_gtw = fs_dgw

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, model_run_id: str = None) -> Word2Vec:
        """Loads a trained spec2vec model given an mlflow_run_id"""
        self.logger.info(f"Loading Spec2Vec model with run_id {model_run_id}.")
        mlflow.set_tracking_uri(MLFLOW_SERVER)
        run: Run = mlflow.get_run(model_run_id)
        model_path = f"{run.info.artifact_uri}/model/python_model.pkl"
        self.logger.info(f"Loading model from path: {model_path}")
        model = self._fs_gtw.read_from_file(model_path)
        return model.model
