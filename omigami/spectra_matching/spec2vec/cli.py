import click

from omigami.cli_options import (
    common_flow_options,
    common_training_options,
    dataset_id,
    ion_mode,
)
from omigami.spectra_matching.spec2vec.main import (
    run_spec2vec_training_flow,
    run_deploy_spec2vec_model_flow,
)
from omigami.utils import add_click_options


@click.group(name="spec2vec")
def spec2vec_cli():
    pass


@spec2vec_cli.command(name="train")
@click.option(
    "--iterations",
    type=int,
    default=25,
    help="Number of iterations of model training",
    show_default=True,
)
@click.option(
    "--n-decimals",
    type=int,
    default=2,
    help="Precision in number of decimals for creating the embeddings",
    show_default=True,
)
@click.option(
    "--window",
    type=int,
    default=500,
    help="Size of window of words context for Word2Vec model",
    show_default=True,
)
@click.option(
    "--intensity-weighting-power",
    type=float,
    default=0.5,
    show_default=True,
    help="Value to elevate the intensities to",
)
@click.option(
    "--allowed-missing-percentage",
    type=float,
    default=5.0,
    show_default=True,
    help="Missing percentage of ions allowed",
)
@add_click_options(common_flow_options)
@add_click_options(common_training_options)
def training_flow_cli(*args, **kwargs):
    run_spec2vec_training_flow(*args, **kwargs)


@spec2vec_cli.command(name="deploy-model")
@click.option(
    "--model-run-id",
    type=str,
    required=True,
    help="Model run ID that will be used to deploy",
)
@click.option(
    "--n-decimals",
    type=int,
    default=2,
    help="Precision in number of decimals for creating the embeddings",
    show_default=True,
)
@click.option(
    "--intensity-weighting-power",
    type=float,
    default=0.5,
    show_default=True,
    help="Value to elevate the intensities to",
)
@click.option(
    "--allowed-missing-percentage",
    type=float,
    default=5.0,
    show_default=True,
    help="Missing percentage of ions allowed",
)
@click.option(
    "--dataset-id",
    type=click.Choice(["small", "10k", "complete"]),
    required=True,
    help="Name of the dataset of choice. It must match the source-uri of the dataset.",
)
@click.option(
    "--ion-mode",
    type=click.Choice(["positive", "negative"]),
    default="positive",
    help="Which ion mode to use.",
    show_default=True,
)
@add_click_options(common_flow_options)
@add_click_options([dataset_id, ion_mode])
def deploy_model_cli(*args, **kwargs):
    run_deploy_spec2vec_model_flow(*args, **kwargs)
