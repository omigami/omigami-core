import pytest
from omigami.ms2deepscore.config import MS2DEEPSCORE_MODEL_URI
from omigami.ms2deepscore.deployment import deploy_minimal_flow
from typing_extensions import Literal
from omigami.config import config, MLFLOW_SERVER


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
        image="drtools/prefect:omigami-SNAPSHOT.bc78dc6",
        environment=env,
        model_uri=MS2DEEPSCORE_MODEL_URI,
        project_name="ms2deepscore",
        mlflow_server=MLFLOW_SERVER,
        flow_name="first-flow-attempt",
        deploy_model=True,
        overwrite=True,
        auth=True,
        **login_config,
    )

    assert flow_id
