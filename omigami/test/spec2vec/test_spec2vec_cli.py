from click.testing import CliRunner

import omigami.spec2vec.cli
from omigami.spec2vec.cli import spec2vec_cli


def spec2vec_cli_assertions(
    image: str,
    project_name: str,
    flow_name: str,
    dataset_id: str,
    source_uri: str,
    ion_mode: str,
    iterations: int,
    n_decimals: int,
    window: int,
    deploy_model: bool,
    dataset_directory: str,
    **kwargs,
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
    assert deploy_model is True
    assert dataset_directory == "dir"
    assert set(kwargs.keys()) == {
        "intensity_weighting_power",
        "allowed_missing_percentage",
        "overwrite_model",
        "overwrite_all_spectra",
    }


def test_spec2vec_cli(monkeypatch):
    monkeypatch.setattr(
        omigami.spec2vec.cli, "run_spec2vec_flow", spec2vec_cli_assertions
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
