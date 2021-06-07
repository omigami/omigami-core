import logging

import prefect
from prefect import Flow

from omigami.data_gateway import InputDataGateway
from omigami.ms2deep.flows.config import FlowConfig
from omigami.ms2deep.tasks import (
    DeployModel,
)

logger = prefect.utilities.logging.get_logger()
logging.basicConfig(level=logging.DEBUG)


def build_training_flow(
    project_name: str,
    input_dgw: InputDataGateway,
    model_output_dir: str,
    flow_config: FlowConfig,
    flow_name: str = "spec2vec-training-flow",
    deploy_model: bool = False,
) -> Flow:
    """TODO"""
    with Flow(flow_name, **flow_config.kwargs) as training_flow:
        if deploy_model:
            DeployModel()(model_registry)

    return training_flow
