from typing import Optional, Tuple, Dict

import pandas as pd

from omigami.authentication.prefect_factory import PrefectClientFactory
from omigami.config import IonModes, API_SERVER_URLS, config
from omigami.spec2vec.deployment import Spec2VecDeployer
from omigami.spec2vec.factory import Spec2VecFlowFactory


def deploy_training_flow(
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
    auth: bool = True,
) -> Tuple[str, str]:
    """
    Builds, deploys, and runs a Spec2Vec model training flow.

    Parameters
    ----------
    For information on parameters please check spec2vec/cli.py

    Returns
    -------
    flow_id, flow_run_id:
        Identifiers for the registered flow and for the run triggered with the flow

    """
    api_server = API_SERVER_URLS[environment]
    login_config = _get_login_config(auth, environment)
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
    flow_id, flow_run_id = deployer.deploy_flow(flow=flow, project_name=project_name)

    return flow_id, flow_run_id


def _get_login_config(auth: bool, environment: str) -> Dict[str, str]:
    if auth:
        login_config = config["login"][environment].get(dict)
        login_config.pop("token")
    else:
        login_config = {
            "username": None,
            "password": None,
            "auth_url": "url",
            "session_token": "token",
        }
    return login_config
