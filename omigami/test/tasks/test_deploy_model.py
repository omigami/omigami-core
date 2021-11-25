import pytest
from prefect import Flow
from prefect.storage import S3

from omigami.spectra_matching.tasks.deploy_model import (
    DeployModelParameters,
    DeployModel,
)


@pytest.mark.skip(reason="This test deploys a seldon model using a model URI.")
def test_deploy_seldon_model(mock_default_config):
    FLOW_CONFIG = {
        "storage": S3("dr-prefect"),
    }
    params = DeployModelParameters("2", "neutral", True, "dev")

    with Flow("debugging-flow", **FLOW_CONFIG) as deploy:
        DeployModel(params)(
            registered_model={
                "model_uri": "s3://omigami-dev/spec2vec/mlflow/tests/f9ba67b8b96040edae87c24f3161da68/artifacts/model/"
            }
        )

    res = deploy.run()

    assert res.is_successful()
