import mlflow
import prefect
import yaml
from kubernetes import client, config
from prefect import task

seldon_deployment = """
    apiVersion: machinelearning.seldon.io/v1alpha2
    kind: SeldonDeployment
    metadata:
      name: spec2vec
    spec:
      name: spec2vec
      predictors:
      - graph:
          children: []
          implementation: MLFLOW_SERVER
          modelUri: dummy
          name: spec2vec
        name: default
        replicas: 1
        traffic: 100
        componentSpecs:
        - spec:
            # We are setting high failureThreshold as installing conda dependencies
            # can take long time and we want to avoid k8s killing the container prematurely
            containers:
              - name: spec2vec
                readinessProbe:
                  failureThreshold: 10
                  initialDelaySeconds: 120
                  periodSeconds: 30
                  successThreshold: 1
                  tcpSocket:
                    port: 9000
                  timeoutSeconds: 3
                livenessProbe:
                  failureThreshold: 10
                  initialDelaySeconds: 120
                  periodSeconds: 30
                  successThreshold: 1
                  tcpSocket:
                    port: 9000
                  timeoutSeconds: 3

"""

CUSTOM_RESOURCE_INFO = dict(
    group="machinelearning.seldon.io",
    version="v1alpha2",
    plural="seldondeployments",
)


@task()
def deploy_model_task(run_id: str, namespace: str = "default"):
    run = mlflow.get_run(run_id)
    model_uri = f"{run.info.artifact_uri}/model/"

    logger = prefect.context.get("logger")

    logger.info(f"Deploying model {model_uri} to enviroment {namespace}")

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    dep = yaml.safe_load(seldon_deployment)
    dep["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

    try:
        resp = custom_api.create_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            body=dep,
        )

        logger.info("Deployment created. status='%s'" % resp["status"]["state"])
    except:
        logger.info("Updating existing model")
        existent_deployment = custom_api.get_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            name=dep["metadata"]["name"],
        )
        existent_deployment["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

        resp = custom_api.replace_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            name=existent_deployment["metadata"]["name"],
            body=existent_deployment,
        )
