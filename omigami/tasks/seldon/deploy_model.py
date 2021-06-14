from pathlib import Path

import yaml
from kubernetes import config, client
from kubernetes.config import ConfigException
from prefect import Task

from omigami.config import SELDON_PARAMS, CLUSTERS
from omigami.helper_classes.exception import DeployingError
from omigami.tasks.config import merge_configs


class DeployModel(Task):
    def __init__(
        self,
        redis_db: str,
        environment: str = "dev",
        overwrite: bool = True,
        ion_mode: str = "positive",
        **kwargs,
    ):
        self._redis_db = redis_db
        self._environment = environment
        self._overwrite = overwrite
        self._ion_mode = ion_mode
        self._seldon_deployment_path = Path(__file__).parent / "seldon_deployment.yaml"
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

        # TODO: WIP. possibly remove these 4 lines below
        # try:
        #     self._create_deployment(custom_api, deployment, overwrite)
        # except:
        #     self._update_deployment(custom_api, deployment, model_uri)

        return resp

    def _create_seldon_deployment(self, model_uri):
        with open(self._seldon_deployment_path) as yaml_file:
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

    def _create_deployment(self, custom_api, deployment, overwrite):
        if overwrite:
            custom_api.delete_namespaced_custom_object(
                **SELDON_PARAMS, name=deployment["metadata"]["name"]
            )
        resp = custom_api.create_namespaced_custom_object(
            **SELDON_PARAMS,
            body=deployment,
        )
        self.logger.info("Deployment created.")

    def _update_deployment(self, custom_api, deployment, model_uri):
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
