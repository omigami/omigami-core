import click
from click.testing import CliRunner

import omigami.spectra_matching.ms2deepscore.cli
from omigami.config import IonModes, STORAGE_ROOT
from omigami.spectra_matching.ms2deepscore.cli import ms2deepscore_cli


def mock_cli(
    image: str,
    project_name: str,
    flow_name: str,
    dataset_id: str,
    source_uri: str,
    ion_mode: IonModes,
    fingerprint_n_bits: int,
    scores_decimals: int,
    spectrum_binner_n_bins: int,
    deploy_model: bool,
    overwrite_model: bool,
    spectrum_ids_chunk_size: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    epochs: int,
    chunk_size: int = 10,
    schedule=None,
    dataset_directory: str = None,
):
    assert image == "image"
    assert project_name == "project name"
    assert flow_name == "flow name"
    assert dataset_id == "10k"
    assert source_uri == "uri"
    assert ion_mode == "positive"
    assert fingerprint_n_bits == 2048
    assert scores_decimals == 5
    assert spectrum_binner_n_bins == 20000
    assert train_ratio == 0.7
    assert validation_ratio == 0.05
    assert test_ratio == 0.05
    assert epochs == 5
    assert chunk_size == 10
    assert spectrum_ids_chunk_size == 10000
    assert deploy_model is False
    assert overwrite_model is False
    assert schedule is None
    assert dataset_directory == str(STORAGE_ROOT / "datasets")


def test_ms2deepscore_cli(monkeypatch):
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.cli,
        "run_ms2deepscore_training_flow",
        mock_cli,
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
        "--fingerprint-n-bits",
        "2048",
        "--spectrum-binner-n-bins",
        "20000",
        "--train-ratio",
        "0.7",
    ]
    runner = CliRunner()

    result = runner.invoke(ms2deepscore_cli, args=cli_command)

    if result.exit_code == 0:
        pass
    elif result.exit_code == 1:
        raise result.exception
    elif result.exit_code == 2:
        raise ValueError(result.output)


def test_ms2ds_deploy_model_cli():
    command = ms2deepscore_cli.commands["deploy-model"]
    required_params = {"model_run_id", "image", "project_name", "flow_name"}
    optional_params = set()
    param_types = [
        click.STRING,
        click.STRING,
        click.STRING,
        click.STRING,
    ]

    assert command.name == "deploy-model"
    assert required_params == {p.name for p in command.params if p.required}
    assert optional_params == {p.name for p in command.params if not p.required}
    assert param_types == [p.type for p in command.params]
