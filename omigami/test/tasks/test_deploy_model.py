import pytest
import yaml
from prefect import Flow
from prefect.storage import S3

from omigami.config import ROOT_DIR
from omigami.tasks.seldon.deploy_model import DeployModel
from omigami.test.conftest import TEST_TASK_CONFIG


@pytest.mark.skip(
    reason="This test is actually deploying to seldon we should change asap"
)
def test_deploy_model_task():

    # TODO: this needs assertions and a way of testing from outside kubernetes environment
    with Flow("test-flow") as test_flow:
        deploy_task = DeployModel(redis_db="2", ion_mode="neutral", overwrite=True)(
            registered_model={"model_uri": "uri"}
        )

        res = test_flow.run()
        data = res.result[deploy_task].result

        assert res.is_successful()

    assert True


def test_create_seldon_deployment():
    task = DeployModel(redis_db="2", ion_mode="neutral")

    deployment = task._create_seldon_deployment("model_uri")

    assert deployment["spec"]["predictors"][0]["graph"]["modelUri"] == "model_uri"
    assert (
        deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][
            0
        ]["env"][-1]["value"]
        == "2"
    )
    assert deployment["metadata"]["name"] == "spec2vec-neutral"
    assert deployment["spec"]["name"] == "spec2vec-neutral"


def test_update_seldon_configs():
    redis_db = "2"
    with Flow("test-flow") as test_flow:
        deploy_task = DeployModel(redis_db=redis_db)(
            registered_model={"model_uri": "uri", "run_id": "1"}, overwrite=False
        )

        seldon_deployment_path = (
            ROOT_DIR / "tasks" / "seldon" / "seldon_deployment.yaml"
        )
        with open(seldon_deployment_path) as yaml_file:
            deployment = yaml.safe_load(yaml_file)
            configs = {"model_uri": "uri"}
            deployment = deploy_task._update_seldon_configs(deployment, configs)
            for kvdict in deployment["spec"]["predictors"][0]["componentSpecs"][0][
                "spec"
            ]["containers"][0]["env"]:
                for key, value in kvdict.items():
                    if key == "REDIS_DB":
                        assert value == redis_db


# @pytest.mark.skip(reason="This test deploys a seldon model using a model URI.")
def test_deploy_seldon_model():
    FLOW_CONFIG = {
        "storage": S3("dr-prefect"),
    }

    with Flow("debugging-flow", **FLOW_CONFIG) as deploy:
        DeployModel(redis_db="0", environment="dev", **TEST_TASK_CONFIG)(
            registered_model={
                "model_uri": "s3://omigami-dev/spec2vec/mlflow/tests/f9ba67b8b96040edae87c24f3161da68/artifacts/model/"
            }
        )

    res = deploy.run()

    assert res.is_successful()
