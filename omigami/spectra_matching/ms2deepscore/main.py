from typing import Optional, Tuple

import pandas as pd

from omigami.authentication.prefect_factory import prefect_client_factory
from omigami.config import IonModes
from omigami.deployer import FlowDeployer
from omigami.spectra_matching.ms2deepscore import MS2DEEPSCORE_PROJECT_NAME
from omigami.spectra_matching.ms2deepscore.factory import MS2DeepScoreFlowFactory
from omigami.spectra_matching.util import run_local_training_flow


def run_ms2deepscore_training_flow(
    image: str,
    flow_name: str,
    dataset_id: str,
    ion_mode: IonModes,
    fingerprint_n_bits: int,
    scores_decimals: int,
    spectrum_binner_n_bins: int,
    spectrum_ids_chunk_size: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    epochs: int,
    project_name: str = MS2DEEPSCORE_PROJECT_NAME,
    schedule: Optional[pd.Timedelta] = None,
    dataset_directory: str = None,
    local: bool = False,
) -> Tuple[str, str]:
    """
    Builds, deploys, and runs a MS2DeepScore model training flow.

    Parameters
    ----------
    For information on parameters please check ms2deepscore/cli.py

    Returns
    -------
    flow_id, flow_run_id:
        Identifiers for the registered flow and for the run triggered with the flow

    """

    factory = MS2DeepScoreFlowFactory(dataset_directory=dataset_directory)
    flow = factory.build_training_flow(
        flow_name=flow_name,
        image=image,
        dataset_id=dataset_id,
        fingerprint_n_bits=fingerprint_n_bits,
        scores_decimals=scores_decimals,
        spectrum_binner_n_bins=spectrum_binner_n_bins,
        ion_mode=ion_mode,
        project_name=project_name,
        spectrum_ids_chunk_size=spectrum_ids_chunk_size,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        epochs=epochs,
        schedule=schedule,
    )
    if local is True:
        flow_run = run_local_training_flow(flow, project_name)
        return flow_run

    deployer = FlowDeployer(prefect_client=prefect_client_factory.get())
    flow_id, flow_run_id = deployer.deploy_flow(flow=flow, project_name=project_name)

    return flow_id, flow_run_id


def run_deploy_ms2ds_model_flow(
    model_run_id: str,
    image: str,
    project_name: str,
    flow_name: str,
    dataset_id: str,
    ion_mode: IonModes,
) -> Tuple[str, str]:
    """
    Builds, deploys, and runs a model deployment flow.

    Parameters
    ----------
    For information on parameters please check ms2deepscore/cli.py

    Returns
    -------
    flow_id, flow_run_id:
        Identifiers for the registered flow and for the run triggered with the flow

    """

    factory = MS2DeepScoreFlowFactory()
    flow = factory.build_model_deployment_flow(
        image=image,
        project_name=project_name,
        flow_name=flow_name,
        dataset_id=dataset_id,
        ion_mode=ion_mode,
    )

    flow_parameters = {"ModelRunID": model_run_id}

    deployer = FlowDeployer(prefect_client=prefect_client_factory.get())
    flow_id, flow_run_id = deployer.deploy_flow(
        flow=flow, project_name=project_name, flow_parameters=flow_parameters
    )

    return flow_id, flow_run_id
