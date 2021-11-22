from unittest.mock import patch

import pytest
from click.testing import CliRunner

from omigami.config import (
    API_SERVER_URLS,
    MLFLOW_SERVER,
    SOURCE_URI_PARTIAL_GNPS,
    S3_BUCKETS,
)
from omigami.spec2vec.cli import cli
from omigami.spec2vec.config import MODEL_DIRECTORIES, PROJECT_NAME


@pytest.fixture
def cli_parameters():
    parameters = dict(
        project_name=PROJECT_NAME,
        mlflow_server=MLFLOW_SERVER,
        model_output_dir=MODEL_DIRECTORIES,
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        output_dir=S3_BUCKETS,
        flow_name="spec2vec-pretrained-flow",
        dataset_name="small",
        dataset_id=None,
        auth=False,
        auth_url=None,
        username=None,
        password=None,
        api_server=API_SERVER_URLS["dev"],
        n_decimals=2,
        iterations=25,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
        overwrite_model=False,
        overwrite_all_spectra=False,
    )
    return parameters


def test_deploy_default_training_flow(monkeypatch, cli_parameters):

    func = "omigami.spec2vec.cli.Spec2VecDeployer"

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

    func = "omigami.spec2vec.cli.Spec2VecDeployer"

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
            ],
        )

    assert result.exit_code == 0

    cli_parameters.pop("flow_name")
    patch_func.assert_called_once_with(image="my-image", **cli_parameters)
