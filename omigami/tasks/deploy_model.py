from dataclasses import dataclass

from prefect import Task

from omigami.config import CLUSTER, SELDON_PARAMS
from omigami.seldon.seldon_deployment import SeldonDeployment
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
        super().__init__(**config)

    def run(self, registered_model: dict = None) -> None:
        sd = SeldonDeployment(context=CLUSTER)

        model_uri = registered_model["model_uri"]
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
