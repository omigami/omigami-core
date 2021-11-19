from typing import Optional

import click
import pandas as pd

from omigami.authentication.prefect_factory import PrefectClientFactory
from omigami.config import (
    API_SERVER_URLS,
    SOURCE_URI_PARTIAL_GNPS,
    IonModes,
    config,
)
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.deployment import Spec2VecDeployer
from omigami.spec2vec.factory import Spec2VecFlowFactory


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
    "--environment",
    "-e",
    type=click.Choice(["dev", "prod"]),
    default="dev",
    help="Which environment to run the flow on.",
    show_default=True,
)
@click.option(
    "--deploy-model",
    is_flag=True,
    type=bool,
    default=False,
    help="Flag to whether deploy the model to seldon or not",
    show_default=True,
)
@click.option(
    "--overwrite-model",
    is_flag=True,
    help="Flag to overwrite existing deployed model.",
    show_default=True,
)
@click.option(
    "--overwrite-all-spectra",
    is_flag=True,
    help="Flag to overwrite all processed spectra and create them again.",
    show_default=True,
)
def training_flow_cli(
    image: str,
    project_name: str,
    flow_name: str,
    dataset_name: str,
    source_uri: str,
    ion_mode: IonModes,
    iterations: int,
    n_decimals: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    environment: str,
    deploy_model: bool,
    overwrite_model: bool,
    overwrite_all_spectra: bool,
    schedule: Optional[pd.Timedelta] = None,
):
    api_server = API_SERVER_URLS[environment]
    login_config = config["login"][environment].get(dict)
    login_config.pop("token")
    prefect_factory = PrefectClientFactory(api_server=api_server, **login_config)
    client = prefect_factory.get_client()

    factory = Spec2VecFlowFactory(environment=environment)
    flow = factory.build_training_flow(
        image=image,
        project_name=project_name,
        flow_name=flow_name,
        dataset_name=dataset_name,
        source_uri=source_uri,
        ion_mode=ion_mode,
        iterations=iterations,
        n_decimals=n_decimals,
        window=window,
        intensity_weighting_power=intensity_weighting_power,
        allowed_missing_percentage=allowed_missing_percentage,
        deploy_model=deploy_model,
        overwrite_model=overwrite_model,
        overwrite_all_spectra=overwrite_all_spectra,
        schedule=schedule,
    )
    deployer = Spec2VecDeployer(client=client)
    deployer.deploy_flow(flow=flow, project_name=project_name)


@spec2vec_cli.command(name="deploy-model")
def deploy_model_cli():
    pass
