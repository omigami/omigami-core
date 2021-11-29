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
    overwrite_all_spectra: bool,
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
    assert overwrite_all_spectra is False
    assert schedule is None
    assert dataset_directory == str(STORAGE_ROOT / "datasets")


def test_ms2deepscore_cli(monkeypatch):
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.cli, "run_ms2deepscore_flow", mock_cli
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
