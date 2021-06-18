from dataclasses import dataclass
from pathlib import Path

import yaml
from kubernetes import config, client
from kubernetes.config import ConfigException
from prefect import Task

from omigami.config import SELDON_PARAMS, CLUSTERS
from omigami.utils import merge_prefect_task_configs
from omigami.spec2vec.helper_classes.exception import DeployingError


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
        self._redis_db = deploy_parameters.redis_db
        self._environment = deploy_parameters.environment
        self._overwrite = deploy_parameters.overwrite
        self._ion_mode = deploy_parameters.ion_mode
        self._model_name = f"spec2vec-{self._ion_mode}"

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, registered_model: dict = None) -> None:
        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config(context=CLUSTERS[self._environment])
        custom_api = client.CustomObjectsApi()

        model_uri = registered_model["model_uri"]
        self.logger.info(
            f"Deploying model '{self._model_name}' from uri {model_uri}. \nUsing environment"
            f" '{self._environment}' and namespace '{SELDON_PARAMS['namespace']}'."
        )

        deployment = self._create_seldon_deployment(model_uri)

        existing_deployments = [
            obj["metadata"]["name"]
            for obj in custom_api.list_namespaced_custom_object(**SELDON_PARAMS)[
                "items"
            ]
        ]
        if self._model_name in existing_deployments:
            if self._overwrite:
                self.logger.info(
                    f"Overwriting existing deployment for model {self._model_name}"
                )
                custom_api.delete_namespaced_custom_object(
                    **SELDON_PARAMS, name=self._model_name
                )
            else:
                self.logger.warning(
                    f"Seldon deployment not updated: deployment named {self._model_name} already exists "
                    f"in the cluster and 'overwrite' DeployModel task parameter was not set to True."
                )
                return

        resp = custom_api.create_namespaced_custom_object(
            **SELDON_PARAMS,
            body=deployment,
        )
        (status,) = custom_api.get_namespaced_custom_object(
            **SELDON_PARAMS, name=self._model_name
        )["status"]["deploymentStatus"].values()
        self.logger.info(f"Finished deployment. Deployment status: {status}")
        return resp

    def _create_seldon_deployment(self, model_uri: str) -> dict:
        seldon_deployment_path = Path(__file__).parent / "seldon_deployment.yaml"
        with open(seldon_deployment_path) as yaml_file:
            deployment = yaml.safe_load(yaml_file)

        try:
            deployment["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

            # sets redis database on seldom container env config
            deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                "containers"
            ][0]["env"].append({"name": "REDIS_DB", "value": self._redis_db})

            # name according to ion mode
            deployment["metadata"]["name"] = self._model_name
            deployment["spec"]["name"] = self._model_name
            deployment["spec"]["predictors"][0]["graph"]["name"] = self._model_name
            deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                "containers"
            ][0]["name"] = self._model_name

            return deployment

        except KeyError:
            raise DeployingError(
                "Couldn't create a deployment because the configuration schema is not correct"
            )
