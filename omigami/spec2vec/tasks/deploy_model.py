from dataclasses import dataclass

from prefect import Task

from omigami.config import SELDON_PARAMS, CLUSTER
from omigami.seldon.seldon_deployment import SeldonDeployment
from omigami.utils import merge_prefect_task_configs


@dataclass
class DeployModelParameters:
    redis_db: str
    ion_mode: str
    overwrite_model: bool


class DeployModel(Task):
    def __init__(
        self,
        deploy_parameters: DeployModelParameters,
        **kwargs,
    ):
        self._redis_db = deploy_parameters.redis_db
        self._overwrite_model = deploy_parameters.overwrite_model
        self._ion_mode = deploy_parameters.ion_mode
        self._model_name = f"spec2vec-{self._ion_mode}"

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
