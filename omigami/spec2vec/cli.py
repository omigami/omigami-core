import click

from omigami.config import (
    API_SERVER_URLS,
    S3_BUCKETS,
    MLFLOW_SERVER,
    SOURCE_URI_PARTIAL_GNPS,
)
from omigami.spec2vec.config import MODEL_DIRECTORIES, PROJECT_NAME
from omigami.spec2vec.deployment import Spec2VecDeployer
from omigami.utils import add_click_options

configuration_options = [
    click.option("--project-name", "-p", default=PROJECT_NAME),
    click.option("--source-uri", default=SOURCE_URI_PARTIAL_GNPS),
    click.option("--output-dir", default=S3_BUCKETS),
    click.option("--flow-name", default="spec2vec-training-flow"),
    click.option("--model-output-dir", default=MODEL_DIRECTORIES),
    click.option("--mlflow-server", default=MLFLOW_SERVER),
    click.option("--overwrite-model", is_flag=True, help="Overwrite existing model"),
    click.option("--overwrite-all-spectra", is_flag=True, help="Overwrite all spectra"),
]

auth_options = [
    click.option(
        "--api-server", default=API_SERVER_URLS["dev"], help="URL to the prefect API"
    ),
    click.option("--auth", default=False, help="Enable authentication"),
    click.option("--auth_url", default=None, help="Kratos Public URI"),
    click.option("--username", default=None, help="Login username"),
    click.option("--password", default=None, help="Login password"),
]


@click.group()
def cli():
    pass


@cli.command(name="register-training-flow")
@click.option("--image", "-i", type=str, required=True)
@click.option("--dataset-name", type=str)
@click.option("--dataset-id", default=None)
@click.option("--n-decimals", type=int, default=2)
@click.option("--iterations", type=int, default=25)
@click.option("--window", type=int, default=500)
@click.option("--intensity-weighting-power", type=float, default=0.5)
@click.option("--allowed-missing-percentage", type=float, default=5.0)
@add_click_options(auth_options)
@add_click_options(configuration_options)
def deploy_training_flow_cli(flow_name, *args, **kwargs):
    deployer = Spec2VecDeployer(*args, **kwargs)
    deployer.deploy_training_flow(flow_name=flow_name)


if __name__ == "__main__":
    cli()
