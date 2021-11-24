import click

from omigami.cli_options import common_training_options
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.main import deploy_training_flow
from omigami.utils import add_click_options


@click.group(name="spec2vec")
def spec2vec_cli():
    pass


@spec2vec_cli.command(name="train")
@click.option(
    "--project-name",
    "-p",
    default=PROJECT_NAME,
    show_default=True,
    help="Name of the project. This is used as identification by Prefect",
)
@click.option(
    "--flow-name",
    default="spec2vec-training-flow",
    show_default=True,
    help="Name of the flow. This is used as identification by Prefect",
)
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
@add_click_options(common_training_options)
def training_flow_cli(*args, **kwargs):
    deploy_training_flow(*args, **kwargs)


@spec2vec_cli.command(name="deploy-model")
def deploy_model_cli():
    pass
