import pytest
from typing_extensions import Literal

from omigami.config import config, MLFLOW_SERVER
from omigami.ms2deepscore.deployment import deploy_minimal_flow


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

    flow_id = deploy_minimal_flow(
        image="drtools/prefect:omigami-SNAPSHOT.d75e0f92345",
        dataset_name="complete",
        environment=env,
        project_name="ms2deepscore-dev",
        mlflow_server=MLFLOW_SERVER,
        flow_name="test-binning-optimization",
        deploy_model=True,
        overwrite=True,
        auth=True,
        **login_config,
    )

    assert flow_id
