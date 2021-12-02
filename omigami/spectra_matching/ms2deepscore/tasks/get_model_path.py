from typing import Dict

import mlflow
from mlflow.entities import Run
from prefect import Task

from omigami.utils import merge_prefect_task_configs


class GetMS2DeepScoreModelPath(Task):
    def __init__(
        self,
        model_registry_uri: str,
        **kwargs,
    ):
        self._model_registry_uri = model_registry_uri

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, model_run_id: str = None) -> Dict[str, str]:
        """Loads a trained spec2vec model given an mlflow_run_id"""
        self.logger.info(
            f"Loading Spec2Vec model with run_id {model_run_id} from server "
            f"{self._model_registry_uri}."
        )
        mlflow.set_tracking_uri(self._model_registry_uri)
        run: Run = mlflow.get_run(model_run_id)
        model_path = f"{run.info.artifact_uri}/model/artifacts/ms2deep_score.hdf5"
        self.logger.info(f"Found model on path: {model_path}")
        return {"ms2deepscore_model_path": model_path}
