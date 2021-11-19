from click.testing import CliRunner

import omigami.spec2vec.cli
from omigami.spec2vec.cli import spec2vec_cli


def mock_cli(
    image: str,
    project_name: str,
    flow_name: str,
    dataset_name: str,
    source_uri: str,
    ion_mode: str,
    iterations: int,
    n_decimals: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    environment: str,
    deploy_model: bool,
    overwrite_model: bool,
    overwrite_all_spectra: bool,
    schedule=None,
):
    assert image == "image"
    assert project_name == "project name"
    assert flow_name == "flow name"
    assert dataset_name == "10k"
    assert source_uri == "uri"
    assert ion_mode == "positive"
    assert iterations == 25
    assert n_decimals == 5
    assert window == 500
    assert intensity_weighting_power == 0.5
    assert allowed_missing_percentage == 5.0
    assert environment == "dev"
    assert deploy_model is True
    assert overwrite_model is False
    assert overwrite_all_spectra is False
    assert schedule is None


def test_spec2vec_cli(monkeypatch):
    monkeypatch.setattr(omigami.spec2vec.cli, "run_training_flow", mock_cli)
    cli_command = [
        "train",
        "--image",
        "image",
        "--project-name",
        "project name",
        "--flow-name",
        "flow name",
        "--dataset-name",
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
        "-e",
        "dev",
        "--deploy-model",
    ]
    runner = CliRunner()

    result = runner.invoke(spec2vec_cli, args=cli_command)

    if result.exit_code == 0:
        pass
    elif result.exit_code == 1:
        raise result.exception
    elif result.exit_code == 2:
        raise ValueError(result.output)