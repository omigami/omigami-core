import inspect

from omigami.spectra_matching.ms2deepscore.cli import ms2deepscore_cli


def test_ms2deepscore_cli():
    from omigami.spectra_matching.ms2deepscore.cli import run_ms2deepscore_training_flow

    main_args = inspect.getfullargspec(run_ms2deepscore_training_flow)
    command = ms2deepscore_cli.commands["train"]
    required_params = {"dataset_id", "image", "project_name", "flow_name"}
    optional_params = {
        "dataset_directory",
        "epochs",
        "fingerprint_n_bits",
        "ion_mode",
        "schedule",
        "scores_decimals",
        "spectrum_binner_n_bins",
        "spectrum_ids_chunk_size",
        "test_ratio",
        "train_ratio",
        "validation_ratio",
    }

    assert command.name == "train"
    assert set(main_args.args) == {p.name for p in command.params}
    assert required_params == {p.name for p in command.params if p.required}
    assert optional_params == {p.name for p in command.params if not p.required}


def test_ms2ds_deploy_model_cli():
    from omigami.spectra_matching.ms2deepscore.cli import run_deploy_ms2ds_model_flow

    main_args = inspect.getfullargspec(run_deploy_ms2ds_model_flow)
    command = ms2deepscore_cli.commands["deploy-model"]
    required_params = {
        "model_run_id",
        "image",
        "project_name",
        "flow_name",
        "dataset_id",
    }
    optional_params = {"ion_mode"}

    assert command.name == "deploy-model"
    assert set(main_args.args) == {p.name for p in command.params}
    assert required_params == {p.name for p in command.params if p.required}
    assert optional_params == {p.name for p in command.params if not p.required}
