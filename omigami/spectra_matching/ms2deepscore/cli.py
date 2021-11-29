import click

from omigami.cli_options import common_training_options
from omigami.spectra_matching.ms2deepscore.config import PROJECT_NAME
from omigami.spectra_matching.ms2deepscore.main import run_ms2deepscore_flow
from omigami.utils import add_click_options


@click.group(name="ms2deepscore")
def ms2deepscore_cli():
    pass


@ms2deepscore_cli.command(name="train")
@click.option(
    "--project-name",
    "-p",
    default=PROJECT_NAME,
    show_default=True,
    help="Name of the project. This is used as identification by Prefect",
)
@click.option(
    "--flow-name",
    default="ms2deepscore-training-flow",
    show_default=True,
    help="Name of the flow. This is used as identification by Prefect",
)
@click.option(
    "--fingerprint-n-bits",
    type=int,
    default=2048,
    help="Number of bits for molecular fingerprints used on calucation of tanimoto scores",
    show_default=True,
)
@click.option(
    "--scores-decimals",
    type=int,
    default=5,
    help="Decimals used on tanimoto scores",
    show_default=True,
)
@click.option(
    "--spectrum-binner-n-bins",
    type=int,
    default=10000,
    help="Number of bins for the spectrum binner",
    show_default=True,
)
@click.option(
    "--spectrum-ids-chunk-size",
    type=int,
    default=10000,
    show_default=True,
    help="Size of chunking in number of spectrum IDs",
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.9,
    show_default=True,
    help="Fraction of the dataset for the training set",
)
@click.option(
    "--validation-ratio",
    type=float,
    default=0.05,
    show_default=True,
    help="Fraction of the dataset for the validation set",
)
@click.option(
    "--test-ratio",
    type=float,
    default=0.05,
    show_default=True,
    help="Fraction of the dataset for the test set",
)
@click.option(
    "--epochs",
    type=float,
    default=5.0,
    show_default=True,
    help="Number of epochs for training the siamese neural network",
)
@add_click_options(common_training_options)
def training_flow_cli(*args, **kwargs):
    run_ms2deepscore_flow(*args, **kwargs)


@ms2deepscore_cli.command(name="deploy-model")
def deploy_model_cli():
    pass