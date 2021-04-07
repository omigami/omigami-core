from prefect import task

from spec2vec_mlops.helper_classes.model_deployer import ModelDeployer


@task()
def deploy_model_task(
    run_id: str, seldon_deployment_path: str, namespace: str = "default"
):
    model_deployer = ModelDeployer()
    model_deployer.deploy_model(run_id, namespace, seldon_deployment_path)
