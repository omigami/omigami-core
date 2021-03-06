from typing import Optional, Tuple

import pandas as pd

from omigami.authentication.prefect_factory import prefect_client_factory
from omigami.config import IonModes
from omigami.deployer import FlowDeployer
from omigami.spectra_matching.spec2vec import SPEC2VEC_PROJECT_NAME
from omigami.spectra_matching.spec2vec.factory import Spec2VecFlowFactory
from omigami.spectra_matching.util import run_local_training_flow


def run_spec2vec_training_flow(
    image: Optional[str],
    flow_name: str,
    dataset_id: str,
    ion_mode: IonModes,
    iterations: int,
    n_decimals: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    schedule: Optional[pd.Timedelta] = None,
    dataset_directory: str = None,
    local: bool = False,
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
    factory = Spec2VecFlowFactory(dataset_directory=dataset_directory)
    flow = factory.build_training_flow(
        image=image,
        project_name=SPEC2VEC_PROJECT_NAME,
        flow_name=flow_name,
        dataset_id=dataset_id,
        ion_mode=ion_mode,
        iterations=iterations,
        n_decimals=n_decimals,
        window=window,
        intensity_weighting_power=intensity_weighting_power,
        allowed_missing_percentage=allowed_missing_percentage,
        schedule=schedule,
    )
    if local is True:
        flow_run = run_local_training_flow(flow, SPEC2VEC_PROJECT_NAME)
        return flow_run

    deployer = FlowDeployer(prefect_client=prefect_client_factory.get())
    flow_id, flow_run_id = deployer.deploy_flow(
        flow=flow, project_name=SPEC2VEC_PROJECT_NAME
    )

    return flow_id, flow_run_id


def run_deploy_spec2vec_model_flow(
    model_run_id: str,
    image: str,
    flow_name: str,
    dataset_id: str,
    ion_mode: IonModes,
    n_decimals: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
) -> Tuple[str, str]:
    """
    Builds, deploys, and runs a model deployment flow.

    Parameters
    ----------
    For information on parameters please check spec2vec/cli.py

    Returns
    -------
    flow_id, flow_run_id:
        Identifiers for the registered flow and for the run triggered with the flow

    """

    factory = Spec2VecFlowFactory()
    flow = factory.build_model_deployment_flow(
        image=image,
        project_name=SPEC2VEC_PROJECT_NAME,
        flow_name=flow_name,
        dataset_id=dataset_id,
        ion_mode=ion_mode,
        n_decimals=n_decimals,
        intensity_weighting_power=intensity_weighting_power,
        allowed_missing_percentage=allowed_missing_percentage,
    )

    flow_parameters = {"ModelRunID": model_run_id}

    deployer = FlowDeployer(prefect_client=prefect_client_factory.get())
    flow_id, flow_run_id = deployer.deploy_flow(
        flow=flow, project_name=SPEC2VEC_PROJECT_NAME, flow_parameters=flow_parameters
    )

    return flow_id, flow_run_id
