import inspect

from omigami.spectra_matching.spec2vec.cli import spec2vec_cli


def test_spec2vec_training_cli():
    from omigami.spectra_matching.spec2vec.cli import run_spec2vec_training_flow

    main_args = inspect.getfullargspec(run_spec2vec_training_flow)
    command = spec2vec_cli.commands["train"]
    required_params = {"dataset_id", "flow_name"}
    optional_params = {
        "allowed_missing_percentage",
        "dataset_directory",
        "intensity_weighting_power",
        "ion_mode",
        "iterations",
        "n_decimals",
        "schedule",
        "window",
        "local",
        "image",
    }

    assert command.name == "train"
    assert set(main_args.args) == {p.name for p in command.params}
    assert required_params == {p.name for p in command.params if p.required}
    assert optional_params == {p.name for p in command.params if not p.required}


def test_spec2vec_deploy_model_cli():
    from omigami.spectra_matching.spec2vec.cli import run_deploy_spec2vec_model_flow

    main_args = inspect.getfullargspec(run_deploy_spec2vec_model_flow)
    command = spec2vec_cli.commands["deploy-model"]
    required_params = {"model_run_id", "flow_name", "dataset_id"}
    optional_params = {
        "intensity_weighting_power",
        "allowed_missing_percentage",
        "n_decimals",
        "ion_mode",
        "image",
    }

    assert command.name == "deploy-model"
    assert set(main_args.args) == {p.name for p in command.params}
    assert required_params == {p.name for p in command.params if p.required}
    assert optional_params == {p.name for p in command.params if not p.required}
