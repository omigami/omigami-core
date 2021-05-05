from prefect import task

from spec2vec_mlops.helper_classes.model_deployer import ModelDeployer
from spec2vec_mlops.tasks.config import DEFAULT_CONFIG


@task(**DEFAULT_CONFIG)
def deploy_model_task(
    run_id: str,
    seldon_deployment_path: str,
):
    model_deployer = ModelDeployer()
    model_deployer.deploy_model(run_id, seldon_deployment_path, overwrite=True)
