from typing import Dict

import mlflow
from gensim.models import Word2Vec
from mlflow.pyfunc import PythonModel
from prefect import Task

from omigami.model_register import MLFlowModelRegister
from omigami.ms2deepscore.predictor import Predictor
from omigami.utils import merge_prefect_task_configs

CONDA_ENV_PATH = "./requirements/environment.frozen.yaml"


class RegisterModel(Task):
    def __init__(
        self,
        experiment_name: str,
        path: str,
        server_uri: str,
        ms2deepscore_model_path: str,
        **kwargs,
    ):
        self._experiment_name = experiment_name
        self._path = path
        self._server_uri = server_uri
        self._ms2deepscore_model_path = ms2deepscore_model_path

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self) -> Dict[str, str]:
        self.logger.info(
            f"Registering model to {self._server_uri} on URI: {self._path}."
        )

        model_register = ModelRegister(self._server_uri)
        run_id = model_register.register_model(
            Predictor(),
            self._experiment_name,
            self._path,
            CONDA_ENV_PATH,
            artifacts={"ms2deepscore_model_path": self._ms2deepscore_model_path},
        )
        run = mlflow.get_run(run_id)
        self.logger.info(f"{run.info}")

        model_uri = f"{run.info.artifact_uri}/model/"
        return {"model_uri": model_uri, "run_id": run_id}


class ModelRegister(MLFlowModelRegister):
    def register_model(
        self,
        model: PythonModel,
        experiment_name: str,
        path: str,
        conda_env_path: str = None,
        artifacts: Dict = None,
        **kwargs,
    ):
        experiment_id = self._get_or_create_experiment_id(experiment_name, path)
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
            run_id = run.info.run_id
            model.set_run_id(run_id)

            self.log_model(
                model,
                experiment_name,
                path=path,
                code_path=["omigami/ms2deepscore"],
                conda_env_path=conda_env_path,
                artifacts=artifacts,
                **kwargs,
            )

            return run_id
