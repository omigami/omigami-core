import click

from spec2vec_mlops.deployment import deploy_training_flow
from spec2vec_mlops import (
    SOURCE_URI_PARTIAL_GNPS,
    API_SERVER,
    PROJECT_NAME,
    OUTPUT_DIR,
    MODEL_DIR,
    MLFLOW_SERVER,
    DATASET_DIR
)
from spec2vec_mlops.utils import add_options


# I'm not sure we will ever want to expose this on the CLI so I am not including these
# parameters right now. If you think we won't need this we can just tweak config_default
# for changes in configuration. Atm it doesn't look these change often
configuration_options = [
    click.option("--project-name", "-p", default=PROJECT_NAME),
    click.option("--source-uri", default=SOURCE_URI_PARTIAL_GNPS),
    click.option("--output-dir", default=OUTPUT_DIR),
    click.option("--model-output-dir", default=MODEL_DIR),
    click.option("--mlflow-server", default=MLFLOW_SERVER),
]

auth_options = [
    click.option(
        "--api-server", default=API_SERVER["remote"], help="URL to the prefect API"
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
@click.option("--dataset-name", type=str, default=DATASET_DIR)
@click.option("--dataset-id", default=None)
@click.option("--n-decimals", type=int, default=2)
@click.option("--iterations", type=int, default=25)
@click.option("--window", type=int, default=500)
@click.option("--intensity-weighting-power", type=float, default=0.5)
@click.option("--allowed-missing-percentage", type=float, default=5.0)
@click.option("--dataset-size", default=None)
@add_options(auth_options)
@add_options(configuration_options)
def deploy_training_flow_cli(*args, **kwargs):
    _ = deploy_training_flow(*args, **kwargs)


@cli.command(name="register-all-flows")
def deploy_model_cli():
    # spec2vec_model_deployment_pipeline_distributed()
    # flow = build_training_flow()
    pass


if __name__ == "__main__":
    cli()
