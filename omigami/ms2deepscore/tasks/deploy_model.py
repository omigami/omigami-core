from dataclasses import dataclass

from prefect import Task

from omigami.config import SELDON_PARAMS, CLUSTERS
from omigami.seldon.seldon_deployment import SeldonDeployment
from omigami.utils import merge_prefect_task_configs


@dataclass
class DeployModelParameters:
    redis_db: str
    ion_mode: str
    overwrite_model: bool
    environment: str = "dev"
    pretrained: bool = False


class DeployModel(Task):
    """
    Prefect task to deploy model to Kubernetes Cluster
    """

    def __init__(
        self,
        deploy_parameters: DeployModelParameters,
        **kwargs,
    ):
        self._redis_db = deploy_parameters.redis_db
        self._environment = deploy_parameters.environment
        self._overwrite_model = deploy_parameters.overwrite_model
        self._ion_mode = deploy_parameters.ion_mode

        if deploy_parameters.pretrained:
            self._model_name = f"pretrained-ms2deepscore-{self._ion_mode}"
        else:
            self._model_name = f"ms2deepscore-{self._ion_mode}"

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, registered_model: dict = None) -> None:
        """
        This task creates a Seldon deployment in the respective environment and deploys
        model.

        Parameters
        ----------
        registered_model: Dict[str, str]
            Dictionary containing registered `model_uri` and `run_id`

        """
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
            redis_db=self._redis_db,
            overwrite_existing=self._overwrite_model,
        )

        self.logger.info(f"Model deployment finished")

        return res
