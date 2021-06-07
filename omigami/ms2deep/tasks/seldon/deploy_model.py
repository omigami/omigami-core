# TODO: this model deployment task could potentially be shared across different tools

from pathlib import Path

import prefect
import yaml
from kubernetes import config, client
from kubernetes.config import ConfigException
from prefect import Task

from omigami.config import SELDON_PARAMS, CLUSTERS
from omigami.utils import merge_configs

logger = prefect.context.get("logger")


class DeployModel(Task):
    def __init__(self, environment: str = "dev", overwrite: bool = True, **kwargs):
        self.environment = environment
        self.overwrite = overwrite
        config = merge_configs(kwargs)
        super().__init__(**config)
        pass

    def run(self, registered_model: dict = None, overwrite: bool = True) -> None:

        model_uri = registered_model["model_uri"]
        logger.info(
            f"Deploying model {model_uri} to environment {self.environment} and "
            f"namespace {SELDON_PARAMS['namespace']}."
        )

        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config(context=CLUSTERS[self.environment])

        custom_api = client.CustomObjectsApi()
        seldon_deployment_path = Path(__file__).parent / "seldon_deployment.yaml"
        with open(seldon_deployment_path) as yaml_file:
            deployment = yaml.safe_load(yaml_file)
            configs = {"model_uri": model_uri}
            deployment = self._update_seldon_configs(deployment, configs)

        try:
            self._create_deployment(custom_api, deployment, overwrite)
        except:
            self._update_deployment(custom_api, deployment, model_uri)

        return

    def _update_seldon_configs(self, deployment_yaml, configs):
        try:
            deployment_yaml["spec"]["predictors"][0]["graph"]["modelUri"] = configs[
                "model_uri"
            ]

            # sets redis database on seldom container env config
            deployment_yaml["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                "containers"
            ][0]["env"].append({"name": "REDIS_DB", "value": self.redis_db})

            return deployment_yaml

        except KeyError:
            raise KeyError(
                "Couldn't create a deployment because the configuration schema is not correct"
            )

    @staticmethod
    def _create_deployment(custom_api, deployment, overwrite):
        if overwrite:
            custom_api.delete_namespaced_custom_object(
                **SELDON_PARAMS, name=deployment["metadata"]["name"]
            )
        resp = custom_api.create_namespaced_custom_object(
            **SELDON_PARAMS,
            body=deployment,
        )
        logger.info("Deployment created.")

    @staticmethod
    def _update_deployment(custom_api, deployment, model_uri):
        logger.info("Updating existing model")
        existent_deployment = custom_api.get_namespaced_custom_object(
            **SELDON_PARAMS,
            name=deployment["metadata"]["name"],
        )
        try:
            existent_deployment["spec"]["predictors"][0]["graph"][
                "modelUri"
            ] = model_uri
        except KeyError:
            raise KeyError(
                "Couldn't update the deployment because the configuration schema is not correct"
            )
        resp = custom_api.replace_namespaced_custom_object(
            **SELDON_PARAMS,
            name=existent_deployment["metadata"]["name"],
            body=existent_deployment,
        )
