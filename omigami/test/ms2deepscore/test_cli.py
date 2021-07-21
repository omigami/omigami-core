import pytest
from click.testing import CliRunner
from mock import patch

from omigami.config import API_SERVER_URLS, PROJECT_NAME, MLFLOW_SERVER
from omigami.ms2deepscore.cli import cli


@pytest.fixture
def cli_parameters():
    parameters = dict(
        project_name=PROJECT_NAME,
        mlflow_server=MLFLOW_SERVER,
        flow_name="ms2deepscore-pretrained-flow",
        dataset_name="small",
        environment="dev",
        deploy_model=False,
        overwrite=False,
        auth=False,
        auth_url=None,
        username=None,
        password=None,
        api_server=API_SERVER_URLS["dev"],
    )
    return parameters


def test_deploy_default_training_flow(monkeypatch, cli_parameters):

    func = "omigami.ms2deepscore.cli.MS2DeepScoreDeployer"

    with patch(func, spec=True, return_value=True) as patch_func:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "register-training-flow",
                "--image",
                "my-image",
                "--dataset-name",
                "small",
            ],
        )

    assert result.exit_code == 0

    cli_parameters.pop("flow_name")
    patch_func.assert_called_once_with(image="my-image", **cli_parameters)


def test_deploy_custom_training_flow(monkeypatch, cli_parameters):
    cli_parameters["mlflow_server"] = "custom-url"
    cli_parameters["environment"] = "prod"
    cli_parameters["deploy_model"] = True

    func = "omigami.ms2deepscore.cli.MS2DeepScoreDeployer"

    with patch(func, spec=True, return_value=True) as patch_func:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "register-training-flow",
                "--image",
                "my-image",
                "--dataset-name",
                "small",
                "--mlflow-server",
                "custom-url",
                "--env",
                "prod",
                "--deploy-model",
            ],
        )

    assert result.exit_code == 0

    cli_parameters.pop("flow_name")
    patch_func.assert_called_once_with(image="my-image", **cli_parameters)
