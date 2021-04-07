import mlflow
import prefect
import yaml
from kubernetes import client, config


logger = prefect.context.get("logger")

CUSTOM_RESOURCE_INFO = dict(
    group="machinelearning.seldon.io",
    version="v1alpha2",
    plural="seldondeployments",
)


class ModelDeployer:
    def deploy_model(self, run_id: str, namespace: str, seldon_deployment_path: str):
        run = mlflow.get_run(run_id)
        model_uri = f"{run.info.artifact_uri}/model/"

        logger.info(f"Deploying model {model_uri} to environment {namespace}")
        config.load_incluster_config()
        custom_api = client.CustomObjectsApi()
        with open(seldon_deployment_path) as yaml_file:
            deployment = yaml.safe_load(yaml_file)
        deployment["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

        try:
            self._create_deployment(custom_api, namespace, deployment)
        except:
            self._update_deployment(custom_api, namespace, deployment, model_uri)

    @staticmethod
    def _create_deployment(custom_api, namespace, deployment):
        resp = custom_api.create_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            body=deployment,
        )
        logger.info("Deployment created. status='%s'" % resp["status"]["state"])

    @staticmethod
    def _update_deployment(custom_api, namespace, deployment, model_uri):
        logger.info("Updating existing model")
        existent_deployment = custom_api.get_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            name=deployment["metadata"]["name"],
        )
        existent_deployment["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

        resp = custom_api.replace_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            name=existent_deployment["metadata"]["name"],
            body=existent_deployment,
        )
