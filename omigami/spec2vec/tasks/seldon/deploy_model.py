from dataclasses import dataclass
from pathlib import Path

import yaml
from kubernetes import config, client
from kubernetes.config import ConfigException
from prefect import Task

from omigami.config import SELDON_PARAMS, CLUSTERS
from omigami.spec2vec.helper_classes.exception import DeployingError
from omigami.spec2vec.tasks.config import merge_configs


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

        config = merge_configs(kwargs)
        super().__init__(**config)

    def run(self, registered_model: dict = None, overwrite: bool = True) -> None:
        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config(context=CLUSTERS[self._environment])
        custom_api = client.CustomObjectsApi()

        model_uri = registered_model["model_uri"]
        self.logger.info(
            f"Deploying model {model_uri} to environment {self._environment} and "
            f"namespace {SELDON_PARAMS['namespace']}."
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
                    f"Did not update the seldon deployment because there is a deployment "
                    f"named {self._model_name} in the cluster."
                )
                return

        resp = custom_api.create_namespaced_custom_object(
            **SELDON_PARAMS,
            body=deployment,
        )
        self.logger.info("Finished deployment. Model status")
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

    def _create_deployment(
        self, custom_api: client.CustomObjectsApi, deployment: dict, overwrite: bool
    ):
        """Not used at the moment."""
        if overwrite:
            custom_api.delete_namespaced_custom_object(
                **SELDON_PARAMS, name=deployment["metadata"]["name"]
            )
        resp = custom_api.create_namespaced_custom_object(
            **SELDON_PARAMS,
            body=deployment,
        )
        self.logger.info("Deployment created.")

    def _update_deployment(
        self, custom_api: client.CustomObjectsApi, deployment: dict, model_uri: str
    ):
        """Not used at the moment."""
        self.logger.info("Updating existing model")
        existent_deployment = custom_api.get_namespaced_custom_object(
            **SELDON_PARAMS,
            name=deployment["metadata"]["name"],
        )
        try:
            existent_deployment["spec"]["predictors"][0]["graph"][
                "modelUri"
            ] = model_uri
        except KeyError:
            raise DeployingError(
                "Couldn't update the deployment because the configuration schema is not correct"
            )
        resp = custom_api.replace_namespaced_custom_object(
            **SELDON_PARAMS,
            name=existent_deployment["metadata"]["name"],
            body=existent_deployment,
        )
