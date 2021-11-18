from typing import Optional

import click
import pandas as pd

from omigami.authentication.prefect_auth import PrefectAuthenticator
from omigami.cli import cli as omigami_cli
from omigami.config import (
    API_SERVER_URLS,
    SOURCE_URI_PARTIAL_GNPS,
    IonModes,
    config,
)
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.deployment import Spec2VecDeployer
from omigami.spec2vec.factory import Spec2VecFlowFactory


@omigami_cli.group(name="spec2vec")
def spec2vec_cli():
    pass


@spec2vec_cli.command(name="deploy")
@click.option("--project-name", "-p", default=PROJECT_NAME)
@click.option("--source-uri", default=SOURCE_URI_PARTIAL_GNPS)
@click.option("--flow-name", default="spec2vec-training-flow")
@click.option("--overwrite-model", is_flag=True, help="Overwrite existing model")
@click.option("--overwrite-all-spectra", is_flag=True, help="Overwrite all spectra")
@click.option("--image", "-i", type=str, required=True)
@click.option("--dataset-name", type=str)
@click.option("--dataset-id", default=None)
@click.option("--n-decimals", type=int, default=2)
@click.option("--iterations", type=int, default=25)
@click.option("--window", type=int, default=500)
@click.option("--intensity-weighting-power", type=float, default=0.5)
@click.option("--allowed-missing-percentage", type=float, default=5.0)
@click.option("--deploy-model", type=bool, default=False)
@click.option("--auth", default=False, help="Enable authentication")
def deploy_training_flow_cli(
    project_name: str,
    flow_name: str,
    image: str,
    iterations,
    n_decimals: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    dataset_name: str,
    environment: str,
    schedule: Optional[pd.Timedelta],
    ion_mode: IonModes,
    source_uri: str,
    auth: bool,
    deploy_model: bool,
):
    api_server = API_SERVER_URLS[environment]
    login_config = config["login"][environment].get(dict)
    login_config.pop("token")
    authenticator = PrefectAuthenticator(
        auth=auth, api_server=api_server, **login_config
    )
    client = authenticator.get_client()

    factory = Spec2VecFlowFactory(environment=environment)
    flow = factory.build_training_flow(
        flow_name=flow_name,
        project_name=project_name,
        iterations=iterations,
        image=image,
        window=window,
        intensity_weighting_power=intensity_weighting_power,
        dataset_name=dataset_name,
        n_decimals=n_decimals,
        schedule=schedule,
        ion_mode=ion_mode,
        allowed_missing_percentage=allowed_missing_percentage,
        source_uri=source_uri,
        deploy_model=deploy_model,
    )
    deployer = Spec2VecDeployer(client=client)
    deployer.deploy_flow(flow=flow)
