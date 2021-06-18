import pytest
from prefect import Flow
from prefect.storage import S3

from omigami.spec2vec.tasks import DeployModel, DeployModelParameters


@pytest.mark.skip(
    reason="This test is actually deploying to seldon we should change asap"
)
def test_deploy_model_task():
    params = DeployModelParameters("2", "neutral", False, "dev")
    # TODO: this needs assertions and a way of testing from outside kubernetes environment
    with Flow("test-flow") as test_flow:
        deploy_task = DeployModel(params)(registered_model={"model_uri": "uri"})

        res = test_flow.run()
        data = res.result[deploy_task].result

        assert res.is_successful()

    assert True


def test_create_seldon_deployment():
    params = DeployModelParameters("2", "neutral", False, "dev")
    task = DeployModel(params)
    model_name = "spec2vec-neutral"

    deployment = task._create_seldon_deployment("model_uri")

    assert deployment["spec"]["predictors"][0]["graph"]["modelUri"] == "model_uri"
    assert (
        deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][
            0
        ]["env"][-1]["value"]
        == "2"
    )
    assert deployment["metadata"]["name"] == model_name
    assert deployment["spec"]["name"] == model_name
    assert deployment["spec"]["predictors"][0]["graph"]["name"] == model_name
    assert (
        deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][
            0
        ]["name"]
        == model_name
    )


@pytest.mark.skip(reason="This test deploys a seldon model using a model URI.")
def test_deploy_seldon_model(mock_default_config):
    FLOW_CONFIG = {
        "storage": S3("dr-prefect"),
    }
    params = DeployModelParameters("2", "neutral", False, "dev")

    with Flow("debugging-flow", **FLOW_CONFIG) as deploy:
        DeployModel(params)(
            registered_model={
                "model_uri": "s3://omigami-dev/spec2vec/mlflow/tests/f9ba67b8b96040edae87c24f3161da68/artifacts/model/"
            }
        )

    res = deploy.run()

    assert res.is_successful()
