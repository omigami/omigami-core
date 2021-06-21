import pytest
from typing_extensions import Literal

from omigami.config import (
    MLFLOW_SERVER,
    config,
    SOURCE_URI_COMPLETE_GNPS,
)
from omigami.ms2deep.deployment import (
    deploy_prediction_flow,
)


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_prediction_flow():
    """
    BE CAREFUL -> DO NOT set `deploy_model=True` and `env="prod"` unless you know exactly
    what you are doing. Also, make sure to check that your source_uri reflects the dataset
    you want to be downloaded from GNPS and that you match it with the dataset_name
    we use internally to represent each dataset.
    """
    env: Literal["dev", "prod"] = "dev"
    login_config = config["login"][env].get(dict)
    login_config.pop("token")

    flow_id = deploy_prediction_flow(
        image="",
        skip_if_exists=True,
        chunk_size=int(1e8),
        environment=env,
        source_uri=SOURCE_URI_COMPLETE_GNPS,
        project_name="spec2vec",
        mlflow_server=MLFLOW_SERVER,
        flow_name="prediction-flow",
        deploy_model=True,
        overwrite=True,
        auth=True,
        **login_config,
    )

    assert flow_id
