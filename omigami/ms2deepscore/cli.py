import click

from omigami.config import (
    API_SERVER_URLS,
    PROJECT_NAME,
    MLFLOW_SERVER,
)
from omigami.ms2deepscore.deployment import deploy_minimal_flow
from omigami.utils import add_click_options

auth_options = [
    click.option(
        "--api-server", default=API_SERVER_URLS["dev"], help="URL to the prefect API"
    ),
    click.option("--auth", default=False, help="Enable authentication"),
    click.option("--auth-url", default=None, help="Kratos Public URI"),
    click.option("--username", default=None, help="Login username"),
    click.option("--password", default=None, help="Login password"),
]

configuration_options = [
    click.option("--project-name", "-p", default=PROJECT_NAME),
    click.option("--mlflow-server", default=MLFLOW_SERVER),
    click.option("--flow-name", default="ms2deepscore-minimal-flow"),
    click.option("--environment", "--env", default="dev"),
    click.option("--deploy-model", is_flag=True),
    click.option("--overwrite", is_flag=True, help="Overwrite existing model"),
]


@click.group()
def cli():
    pass


@cli.command(name="register-training-flow")
@click.option("--image", "-i", type=str, required=True)
@click.option("--dataset-name", "-d", type=str, required=True)
@add_click_options(auth_options)
@add_click_options(configuration_options)
def deploy_training_flow_cli(*args, **kwargs):
    deploy_minimal_flow(*args, **kwargs)


if __name__ == "__main__":
    cli()
