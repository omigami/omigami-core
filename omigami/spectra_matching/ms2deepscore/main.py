from typing import Optional, Tuple

import pandas as pd

from omigami.authentication.prefect_factory import PrefectClientFactory
from omigami.config import (
    IonModes,
    API_SERVER_URLS,
    OMIGAMI_ENV,
    get_login_config,
    CHUNK_SIZE,
)
from omigami.deployer import FlowDeployer
from omigami.spectra_matching.ms2deepscore.factory import MS2DeepScoreFlowFactory


def run_ms2deepscore_flow(
    image: str,
    project_name: str,
    flow_name: str,
    dataset_id: str,
    source_uri: str,
    ion_mode: IonModes,
    fingerprint_n_bits: int,
    scores_decimals: int,
    spectrum_binner_n_bins: int,
    deploy_model: bool,
    overwrite_model: bool,
    overwrite_all_spectra: bool,
    spectrum_ids_chunk_size: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    epochs: int,
    chunk_size: int = CHUNK_SIZE,
    schedule: Optional[pd.Timedelta] = None,
    dataset_directory: str = None,
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
    api_server = API_SERVER_URLS[OMIGAMI_ENV]
    login_config = get_login_config()
    prefect_factory = PrefectClientFactory(api_server=api_server, **login_config)
    prefect_client = prefect_factory.get_client()

    factory = MS2DeepScoreFlowFactory(dataset_directory=dataset_directory)
    flow = factory.build_training_flow(
        flow_name=flow_name,
        image=image,
        dataset_id=dataset_id,
        fingerprint_n_bits=fingerprint_n_bits,
        scores_decimals=scores_decimals,
        source_uri=source_uri,
        spectrum_binner_n_bins=spectrum_binner_n_bins,
        ion_mode=ion_mode,
        deploy_model=deploy_model,
        overwrite_all_spectra=overwrite_all_spectra,
        overwrite_model=overwrite_model,
        project_name=project_name,
        spectrum_ids_chunk_size=spectrum_ids_chunk_size,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        epochs=epochs,
        chunk_size=chunk_size,
        schedule=schedule,
    )
    deployer = FlowDeployer(prefect_client=prefect_client)
    flow_id, flow_run_id = deployer.deploy_flow(flow=flow, project_name=project_name)

    return flow_id, flow_run_id
