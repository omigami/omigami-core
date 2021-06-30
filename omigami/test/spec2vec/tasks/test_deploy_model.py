import pytest
from prefect import Flow
from prefect.storage import S3

from omigami.spec2vec.tasks import DeployModel, DeployModelParameters


@pytest.mark.skip(
    reason="This test is actually deploying to seldon we should change asap"
)
def test_deploy_model_task():
    params = DeployModelParameters(
        redis_db="2", ion_mode="neutral", overwrite=True, environment="dev"
    )
    with Flow("test-flow") as test_flow:
        deploy_task = DeployModel(params)(registered_model={"model_uri": "uri"})

        res = test_flow.run()
        data = res.result[deploy_task].result

        assert res.is_successful()

    assert True


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
