import click

from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS,
    API_SERVER_URLS,
    PROJECT_NAME,
    S3_BUCKETS,
    MODEL_DIRECTORIES,
    MLFLOW_SERVER,
)
from omigami.ms2deep.deployment import deploy_training_flow
from omigami.utils import add_options

configuration_options = [
    click.option("--project-name", "-p", default=PROJECT_NAME),
    click.option("--source-uri", default=SOURCE_URI_PARTIAL_GNPS),
    click.option("--output-dir", default=S3_BUCKETS),
    click.option("--model-output-dir", default=MODEL_DIRECTORIES),
    click.option("--mlflow-server", default=MLFLOW_SERVER),
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
@add_options(auth_options)
@add_options(configuration_options)
def deploy_training_flow_cli(*args, **kwargs):
    _ = deploy_training_flow(*args, **kwargs)


if __name__ == "__main__":
    cli()
