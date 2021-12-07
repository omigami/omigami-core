from dataclasses import dataclass

import mlflow
from prefect import Task
from prefect.triggers import all_successful

from omigami.config import CLUSTER, SELDON_PARAMS
from omigami.spectra_matching.seldon.seldon_deployment import SeldonDeployment
from omigami.utils import merge_prefect_task_configs


@dataclass
class DeployModelParameters:
    redis_db: str
    overwrite_model: bool
    model_name: str


class DeployModel(Task):
    def __init__(
        self,
        deploy_parameters: DeployModelParameters,
        **kwargs,
    ):
        self._redis_db = deploy_parameters.redis_db
        self._overwrite_model = deploy_parameters.overwrite_model
        self._model_name = deploy_parameters.model_name

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=all_successful)

    def run(self, model_run_id: str = None) -> None:
        """
        Prefect task to deploy model to Kubernetes Cluster

        Parameters
        ----------
        model_run_id:
            Registered model's `run_id`

        """
        sd = SeldonDeployment(context=CLUSTER)

        artifact_uri = mlflow.get_run(model_run_id).info.artifact_uri
        model_uri = f"{artifact_uri}/model/"
        self.logger.info(
            f"Deploying model '{self._model_name}' from uri {model_uri}. \nUsing cluster"
            f" '{CLUSTER}' and namespace '{SELDON_PARAMS['namespace']}'."
        )

        res, status = sd.deploy_model(
            model_name=self._model_name,
            model_uri=model_uri,
            redis_db=self._redis_db,
            overwrite_existing=self._overwrite_model,
        )

        self.logger.info(f"Model deployment finished")

        return res
