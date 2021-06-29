import pytest
from typing_extensions import Literal


from omigami.config import config, MLFLOW_SERVER

from omigami.spec2vec.config import SOURCE_URI_PARTIAL_GNPS
from omigami.spec2vec.deployment import (
    deploy_training_flow,
)


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_training_flow():
    """
    BE CAREFUL -> DO NOT set `deploy_model=True` and `env="prod"` unless you know exactly
    what you are doing. Also, make sure to check that your source_uri reflects the dataset
    you want to be downloaded from GNPS and that you match it with the dataset_name
    we use internally to represent each dataset.
    """
    env: Literal["dev", "prod"] = "dev"
    login_config = config["login"][env].get(dict)
    login_config.pop("token")

    flow_id = deploy_training_flow(
        image="drtools/prefect:omigami-SNAPSHOT.e568c792",
        iterations=15,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        n_decimals=2,
        skip_if_exists=True,
        chunk_size=int(1e8),
        environment=env,
        ion_mode="positive",
        dataset_name="small",
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        project_name="spec2vec",
        mlflow_server=MLFLOW_SERVER,
        flow_name="training-flow/positive",
        deploy_model=True,
        overwrite=True,
        auth=True,
        **login_config,
    )

    assert flow_id
