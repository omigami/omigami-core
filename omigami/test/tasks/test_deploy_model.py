import pytest
import yaml
from prefect import Flow

from omigami.config import ROOT_DIR
from omigami.tasks.seldon.deploy_model import DeployModelTask


@pytest.mark.skip(
    reason="This test is actually deploying to seldon we should change asap"
)
def test_deploy_model_task():

    # TODO: this needs assertions and a way of testing from outside kubernetes environment
    with Flow("test-flow") as test_flow:
        deploy_task = DeployModelTask(redis_db="2")(
            registered_model={"model_uri": "uri", "run_id": "1"}, overwrite=False
        )

        res = test_flow.run()
        data = res.result[deploy_task].result

        assert res.is_successful()

    assert True


def test_update_seldon_configs():
    redis_db = "2"
    with Flow("test-flow") as test_flow:
        deploy_task = DeployModelTask(redis_db=redis_db)(
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
