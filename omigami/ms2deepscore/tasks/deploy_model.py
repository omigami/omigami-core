from dataclasses import dataclass

from prefect import Task

from omigami.config import SELDON_PARAMS, CLUSTERS
from omigami.seldon.seldon_deployment import SeldonDeployment
from omigami.utils import merge_prefect_task_configs


@dataclass
class DeployModelParameters:
    overwrite: bool
    environment: str = "dev"


class DeployModel(Task):
    """
    Prefect task to deploy model to kubernetes
    """

    def __init__(
        self,
        deploy_parameters: DeployModelParameters,
        **kwargs,
    ):
        self._environment = deploy_parameters.environment
        self._overwrite = deploy_parameters.overwrite
        self._model_name = f"ms2deepscore"

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

        res, status = sd.deploy_model(
            model_name=self._model_name,
            model_uri=model_uri,
            overwrite_existing=self._overwrite,
        )

        self.logger.info(f"Model deployment finished")

        return res
