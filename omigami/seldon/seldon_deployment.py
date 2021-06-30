from logging import getLogger
from pathlib import Path

import yaml
from kubernetes import config, client
from kubernetes.config import ConfigException

from omigami.config import SELDON_PARAMS
from omigami.spec2vec.helper_classes.exception import SeldonDeploymentError

DEPLOYMENT_SPEC_PATH = Path(__file__).parent / "seldon_deployment.yaml"

log = getLogger(__name__)


class SeldonDeployment:
    def __init__(
        self,
        context: str = "omigami-dev",
        deployment_spec_path: str = DEPLOYMENT_SPEC_PATH,
        seldon_config: dict = SELDON_PARAMS,
    ):
        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config(context=context)

        self.client = client.CustomObjectsApi()
        self.deployment_spec = self._load_deployment_spec(deployment_spec_path)
        self.seldon_config = seldon_config

    @staticmethod
    def _load_deployment_spec(deployment_spec_path: str):
        """Load deployment spec from specified path"""
        with open(deployment_spec_path) as yaml_file:
            deployment = yaml.safe_load(yaml_file)
            return deployment

    def _delete_existing_deployment(self, deployment_name: str):
        """Delete existing deployment"""
        log.info(f"Removing model deployment {deployment_name}")
        self.client.delete_namespaced_custom_object(
            name=deployment_name, **self.seldon_config
        )

    def deploy_model(
        self,
        model_name: str,
        model_uri: str,
        redis_db: str = "",
        overwrite_existing: bool = False,
    ) -> tuple:
        """
        Create a Custom Seldon Deployment to specified namespace

        Parameters
        ----------
        model_name: Name of the model to be deployed
        model_uri: URI of the model
        redis_db: redis db address
        overwrite_existing: whether to overwrite existing model

        Returns
        -------
            Reference to the model deployment and status of deployment
        """
        existing_deployments = self.list_deployments()

        if model_name in existing_deployments:
            if overwrite_existing:
                log.info(f"Overwriting existing deployment for model {model_name}")
                self._delete_existing_deployment(model_name)
            else:
                raise SeldonDeploymentError(
                    f"Seldon deployment not updated: deployment named {model_name} already exists "
                    f"in the cluster and 'overwrite' DeployModel task parameter was not set to True."
                )

        deployment_spec = self.config_deployment_spec(model_name, model_uri, redis_db)

        log.info(f"Deploying model {model_name} from uri {model_uri}.")
        res = self.client.create_namespaced_custom_object(
            **self.seldon_config,
            body=deployment_spec,
        )

        status = None
        while status is None:
            status = self.client.get_namespaced_custom_object_status(
                **self.seldon_config,
                name=model_name,
            )
            log.debug(f"deployment crd status  {status}")

            if "status" in status:
                status = status["status"]["state"]

        log.info(f"Finished deployment. Deployment status: {status}")

        return res, status

    def config_deployment_spec(self, model_name: str, model_uri: str, redis_db: str):
        """Config the deployment specification with the model data"""
        deployment = self.deployment_spec

        try:
            deployment["metadata"]["namespace"] = self.seldon_config["namespace"]
            deployment["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

            # sets redis database on seldom container env config
            deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                "containers"
            ][0]["env"].append({"name": "REDIS_DB", "value": redis_db})

            # name according to ion mode
            deployment["metadata"]["name"] = model_name
            deployment["spec"]["name"] = model_name
            deployment["spec"]["predictors"][0]["graph"]["name"] = model_name
            deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"][
                "containers"
            ][0]["name"] = model_name

            return deployment

        except KeyError:
            raise SeldonDeploymentError(
                "Unable to config the model deployment spec due to error in the schema"
            )

    def list_deployments(self) -> list:
        """List all deployments in the seldon namespace"""
        return [
            obj["metadata"]["name"]
            for obj in self.client.list_namespaced_custom_object(**self.seldon_config)[
                "items"
            ]
        ]
