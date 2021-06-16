import pytest
from typing_extensions import Literal


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
    # -- setup env --
    # BE CAREFUL -> DO NOT set deploy_model=True and env="prod" unless you
    # know exactly what you are doing.
    # also, make sure to check that your source_uri reflects the dataset
    # you want to be downloaded from GNPS and that you match it with the
    # dataset_name monicker we use internally to represent each dataset.
    env: Literal["dev", "prod"] = "prod"
    login_config = config["login"][env].get(dict)
    login_config.pop("token")

    flow_id = deploy_training_flow(
        image="drtools/prefect:omigami-SNAPSHOT.bc19d2b",
        iterations=3,
        window=300,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        n_decimals=2,
        skip_if_exists=True,
        chunk_size=int(1e8),
        environment=env,
        ion_mode="negative",
        dataset_name="complete",
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        project_name="spec2vec-test",
        mlflow_server=MLFLOW_SERVER,
        flow_name="training-flow/chunk-by-ion-mode",
        deploy_model=True,
        auth=True,
        **login_config,
    )

    assert flow_id
