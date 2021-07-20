import pytest
from typing_extensions import Literal

from omigami.config import config, MLFLOW_SERVER
from omigami.ms2deepscore.deployment import Deployer


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_minimal_flow():
    """
    BE CAREFUL -> DO NOT set `deploy_model=True` and `env="prod"` unless you know exactly
    what you are doing.
    """
    env: Literal["dev", "prod"] = "dev"
    login_config = config["login"][env].get(dict)
    login_config.pop("token")

    deployer = Deployer(
        image="drtools/prefect:omigami-SNAPSHOT.f70e5c2",
        dataset_name="small",
        environment=env,
        project_name="ms2deepscore-dev",
        mlflow_server=MLFLOW_SERVER,
        deploy_model=True,
        overwrite=True,
        auth=True,
        **login_config,
    )
    flow_id = deployer.deploy_minimal_flow(flow_name="running-dev")

    assert flow_id


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_training_flow():
    """
    BE CAREFUL -> DO NOT set `deploy_model=True` and `env="prod"` unless you know exactly
    what you are doing.
    """
    env: Literal["dev", "prod"] = "dev"
    login_config = config["login"][env].get(dict)
    login_config.pop("token")

    deployer = Deployer(
        image="drtools/prefect:omigami-SNAPSHOT.f70e5c2",
        dataset_name="small",
        environment=env,
        project_name="ms2deepscore-dev",
        mlflow_server=MLFLOW_SERVER,
        auth=True,
        **login_config,
    )
    flow_id = deployer.deploy_training_flow(flow_name="running-dev")

    assert flow_id
