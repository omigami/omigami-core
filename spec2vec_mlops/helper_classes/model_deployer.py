import mlflow
import prefect
import yaml
from kubernetes import client, config

from spec2vec_mlops import config as spec2vec_config
from spec2vec_mlops.helper_classes.exception import DeployingError

logger = prefect.context.get("logger")
CUSTOM_RESOURCE_INFO = spec2vec_config["k8s"]["custom_seldon_resource"]


class ModelDeployer:
    def deploy_model(self, run_id: str, seldon_deployment_path: str):
        run = mlflow.get_run(run_id)
        model_uri = f"{run.info.artifact_uri}/model/"

        logger.info(
            f"Deploying model {model_uri} to environment {CUSTOM_RESOURCE_INFO['namespace']}"
        )
        config.load_incluster_config()
        custom_api = client.CustomObjectsApi()
        with open(seldon_deployment_path) as yaml_file:
            deployment = yaml.safe_load(yaml_file)
        try:
            deployment["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri
        except KeyError:
            raise DeployingError(
                "Couldn't create a deployment because the configuration schema is not correct"
            )
        try:
            self._create_deployment(custom_api, deployment)
        except:
            self._update_deployment(custom_api, deployment, model_uri)

    @staticmethod
    def _create_deployment(custom_api, deployment):
        resp = custom_api.create_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            body=deployment,
        )
        logger.info("Deployment created. status='%s'" % resp["status"]["state"])

    @staticmethod
    def _update_deployment(custom_api, deployment, model_uri):
        logger.info("Updating existing model")
        existent_deployment = custom_api.get_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
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
            **CUSTOM_RESOURCE_INFO,
            name=existent_deployment["metadata"]["name"],
            body=existent_deployment,
        )
