import pytest

from omigami.config import (
    MLFLOW_SERVER,
    config,
    SOURCE_URI_PARTIAL_GNPS,
)
from omigami.deployment import (
    deploy_training_flow,
)


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_training_flow():
    login_config = config["login"]["dev"].get(dict)
    login_config.pop("token")
    flow_id = deploy_training_flow(
        image="drtools/prefect:omigami-SNAPSHOT.4fd6f68",
        iterations=5,
        window=300,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        n_decimals=2,
        skip_if_exists=True,
        chunk_size=int(1e8),
        environment="dev",
        dev_dataset_name="small",
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        project_name="spec2vec-test",
        mlflow_server=MLFLOW_SERVER,
        flow_name="training-flow/full-chunked",
        deploy_model=False,
        auth=True,
        auth_url=config["auth_url"].get(str),
        **login_config,
    )

    assert flow_id
