from typing import Optional, Tuple

import pandas as pd

from omigami.authentication.prefect_factory import PrefectClientFactory
from omigami.config import IonModes, API_SERVER_URLS, OMIGAMI_ENV, get_login_config
from omigami.deployer import FlowDeployer
from omigami.spectra_matching.spec2vec.factory import Spec2VecFlowFactory


def run_spec2vec_flow(
    image: str,
    project_name: str,
    flow_name: str,
    dataset_id: str,
    source_uri: str,
    ion_mode: IonModes,
    iterations: int,
    n_decimals: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    deploy_model: bool,
    overwrite_model: bool,
    overwrite_all_spectra: bool,
    schedule: Optional[pd.Timedelta] = None,
    dataset_directory: str = None,
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
    api_server = API_SERVER_URLS[OMIGAMI_ENV]
    login_config = get_login_config()
    prefect_factory = PrefectClientFactory(api_server=api_server, **login_config)
    prefect_client = prefect_factory.get_client()

    factory = Spec2VecFlowFactory(dataset_directory=dataset_directory)
    flow = factory.build_spec2vec_flow(
        image=image,
        project_name=project_name,
        flow_name=flow_name,
        dataset_id=dataset_id,
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
    deployer = FlowDeployer(prefect_client=prefect_client)
    flow_id, flow_run_id = deployer.deploy_flow(flow=flow, project_name=project_name)

    return flow_id, flow_run_id
