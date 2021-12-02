import click
from click.testing import CliRunner

import omigami.spectra_matching.spec2vec.cli
from omigami.spectra_matching.spec2vec.cli import spec2vec_cli


def mock_cli(
    image: str,
    project_name: str,
    flow_name: str,
    dataset_id: str,
    source_uri: str,
    ion_mode: str,
    iterations: int,
    n_decimals: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    deploy_model: bool,
    overwrite_model: bool,
    overwrite_all_spectra: bool,
    dataset_directory: str,
    schedule=None,
):
    assert image == "image"
    assert project_name == "project name"
    assert flow_name == "flow name"
    assert dataset_id == "10k"
    assert source_uri == "uri"
    assert ion_mode == "positive"
    assert iterations == 25
    assert n_decimals == 5
    assert window == 500
    assert intensity_weighting_power == 0.5
    assert allowed_missing_percentage == 5.0
    assert deploy_model is True
    assert overwrite_model is False
    assert overwrite_all_spectra is False
    assert schedule is None
    assert dataset_directory == "dir"


def test_spec2vec_training_cli(monkeypatch):
    monkeypatch.setattr(
        omigami.spectra_matching.spec2vec.cli, "run_spec2vec_training_flow", mock_cli
    )
    cli_command = [
        "train",
        "--image",
        "image",
        "--project-name",
        "project name",
        "--flow-name",
        "flow name",
        "--dataset-id",
        "10k",
        "--source-uri",
        "uri",
        "--ion-mode",
        "positive",
        "--iterations",
        "25",
        "--n-decimals",
        "5",
        "--window",
        "500",
        "--deploy-model",
        "--dataset-directory",
        "dir",
    ]
    runner = CliRunner()

    result = runner.invoke(spec2vec_cli, args=cli_command)

    if result.exit_code == 0:
        pass
    elif result.exit_code == 1:
        raise result.exception
    elif result.exit_code == 2:
        raise ValueError(result.output)


def test_spec2vec_deploy_model_cli():
    command = spec2vec_cli.commands["deploy-model"]
    required_params = {"model_run_id", "image", "project_name", "flow_name"}
    optional_params = {
        "intensity_weighting_power",
        "allowed_missing_percentage",
        "n_decimals",
    }
    param_types = [
        click.STRING,
        click.INT,
        click.FLOAT,
        click.FLOAT,
        click.STRING,
        click.STRING,
        click.STRING,
    ]

    assert command.name == "deploy-model"
    assert required_params == {p.name for p in command.params if p.required}
    assert optional_params == {p.name for p in command.params if not p.required}
    assert param_types == [p.type for p in command.params]
