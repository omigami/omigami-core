import click

from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS,
)
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.main import deploy_training_flow


@click.group(name="spec2vec")
def spec2vec_cli():
    pass


@spec2vec_cli.command(name="train")
@click.option(
    "--image", "-i", type=str, required=True, help="Image used to build the flow"
)
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
    "--dataset-name",
    type=click.Choice(["small", "10k", "complete"]),
    required=True,
    help="Name of the dataset of choice.",
)
@click.option(
    "--source-uri",
    default=SOURCE_URI_PARTIAL_GNPS,
    help="URI to download training data from. Only downloads if it is not already downloaded",
)
@click.option(
    "--ion-mode",
    type=click.Choice(["positive", "negative"]),
    default="positive",
    help="Which ion mode to use.",
    show_default=True,
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
    help="Window to the soul of the compound",
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
    "--deploy-model",
    is_flag=True,
    type=bool,
    default=False,
    help="Flag to whether deploy the model as a seldon deployment or not",
    show_default=True,
)
@click.option(
    "--overwrite-model",
    is_flag=True,
    help="Flag to overwrite existing deployed model. Only works with --deploy-model flag",
    show_default=True,
    default=False,
)
@click.option(
    "--overwrite-all-spectra",
    is_flag=True,
    help="Flag to overwrite all processed spectra and create them again.",
    show_default=True,
    default=False,
)
def training_flow_cli(*args, **kwargs):
    deploy_training_flow(*args, **kwargs)


@spec2vec_cli.command(name="deploy-model")
def deploy_model_cli():
    pass
