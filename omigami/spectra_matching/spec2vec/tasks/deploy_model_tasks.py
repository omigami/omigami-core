from typing import List

import mlflow
from prefect import Task

from omigami.config import MLFLOW_SERVER
from omigami.spectra_matching.storage import DataGateway
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
        """TODO"""
        document_paths = self._fs_gtw.list_files(self._documents_directory)
        chunked_paths = [
            document_paths[i * self._chunk_size : (i + 1) * self._chunk_size]
            for i in range(len(document_paths) // self._chunk_size + 1)
        ]
        return chunked_paths


class LoadSpec2VecModel(Task):
    def __init__(
        self,
        model_id: str,
        fs_dgw: DataGateway,
        **kwargs,
    ):
        self._model_id = model_id
        self._fs_gtw = fs_dgw

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self) -> List[str]:
        """TODO"""
        mlflow.set_tracking_uri(MLFLOW_SERVER)
        self._fs_gtw.read_from_file(model_path)
        pass
