from prefect import Task
from dataclasses import dataclass
from omigami.config import SELDON_PARAMS, CLUSTERS
from omigami.seldon.seldon_deployment import SeldonDeployment
from omigami.utils import merge_prefect_task_configs


@dataclass
class DeployModelParameters:
    redis_db: str
    ion_mode: str
    overwrite: bool
    environment: str = "dev"


class DeployModel(Task):
    def __init__(
        self,
        deploy_parameters: DeployModelParameters,
        **kwargs,
    ):
        self._environment = deploy_parameters.environment
        self._overwrite = deploy_parameters.overwrite
        self._model_name = f"ms2deepscore-{self._ion_mode}"

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, registered_model: dict = None) -> None:
        sd = SeldonDeployment(
            context=CLUSTERS[self._environment],
        )

        model_uri = registered_model["model_uri"]
        self.logger.info(
            f"Deploying model '{self._model_name}' from uri {model_uri}. \nUsing environment"
            f" '{self._environment}' and namespace '{SELDON_PARAMS['namespace']}'."
        )

        res = sd.deploy_model(
            model_name=self._model_name,
            model_uri=model_uri,
            redis_db=self._redis_db,
            overwrite_existing=self._overwrite,
        )

        self.logger.info(f"Model deployment finished")

        return res
