from typing import Dict

import mlflow
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
        mlflow_output_path: str,
        server_uri: str,
        **kwargs,
    ):
        self._experiment_name = experiment_name
        self._mlflow_output_path = mlflow_output_path
        self._server_uri = server_uri

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, model_path: str) -> Dict[str, str]:
        self.logger.info(
            f"Registering model to {self._server_uri} on URI: {self._mlflow_output_path}."
        )

        model_register = ModelRegister(self._server_uri)
        run_id = model_register.register_model(
            Predictor(),
            self._experiment_name,
            self._mlflow_output_path,
            CONDA_ENV_PATH,
            artifacts={"ms2deepscore_model_path": model_path},
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
        output_path: str,
        conda_env_path: str = None,
        artifacts: Dict = None,
        **kwargs,
    ):
        experiment_id = self._get_or_create_experiment_id(experiment_name, output_path)
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
            run_id = run.info.run_id
            model.set_run_id(run_id)

            self.log_model(
                model,
                experiment_name,
                output_path=output_path,
                code_path=[
                    "omigami",
                ],
                conda_env_path=conda_env_path,
                artifacts=artifacts,
                **kwargs,
            )

            return run_id
