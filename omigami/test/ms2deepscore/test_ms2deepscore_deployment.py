import pytest
from typing_extensions import Literal

from omigami.config import (
    config,
    MLFLOW_SERVER,
    SOURCE_URI_COMPLETE_GNPS,
)
from omigami.ms2deepscore.deployment import MS2DeepScoreDeployer


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_pretrained_flow():
    """
    BE CAREFUL -> DO NOT set `deploy_model=True` and `env="prod"` unless you know exactly
    what you are doing.
    """
    env: Literal["dev", "prod"] = "dev"
    login_config = config["login"][env].get(dict)
    login_config.pop("token")

    deployer = MS2DeepScoreDeployer(
        image="drtools/prefect:omigami-SNAPSHOT.1180d27",
        dataset_name="small",
        environment=env,
        project_name="ms2deepscore-dev",
        mlflow_server=MLFLOW_SERVER,
        deploy_model=True,
        overwrite_model=True,
        auth=True,
        overwrite_all_spectra=True,
        **login_config,
    )
    flow_id = deployer.deploy_pretrained_flow(flow_name="running-dev")

    assert flow_id


# @pytest.mark.skip(
#     reason="This test uses internet connection and deploys a test flow to prefect."
# )
def test_deploy_training_flow():
    """
    BE CAREFUL -> DO NOT set `deploy_model=True` and `env="prod"` unless you know exactly
    what you are doing.
    """
    env: Literal["dev", "prod"] = "prod"
    login_config = config["login"][env].get(dict)
    # login_config.pop("token")

    deployer = MS2DeepScoreDeployer(
        image="drtools/prefect:omigami-SNAPSHOT.62b75aa9",
        dataset_name="complete",  # ms2deepscore can not be trained with the small dataset
        # because the minimum batch size to train is 32 samples and the small dataset
        # will lead to less samples than that.
        environment=env,
        project_name="ms2deepscore",
        mlflow_server=MLFLOW_SERVER,
        source_uri=SOURCE_URI_COMPLETE_GNPS,
        auth=True,
        overwrite_all_spectra=False,
        **login_config,
        deploy_model=True,
        overwrite_model=True,
        ion_mode="positive",
    )
    flow_id = deployer.deploy_training_flow(flow_name="make-embeddings")

    assert flow_id
