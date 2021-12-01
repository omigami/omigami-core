from typing import Union

from prefect import Flow, unmapped, Parameter

from omigami.config import IonModes, ION_MODES
from omigami.flow_config import FlowConfig
from omigami.spectra_matching.spec2vec.tasks import (
    MakeEmbeddings,
    MakeEmbeddingsParameters,
)
from omigami.spectra_matching.spec2vec.tasks.deploy_model_tasks import (
    ListDocumentPaths,
    LoadSpec2VecModel,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway, FSDataGateway
from omigami.spectra_matching.tasks import (
    DeployModelParameters,
    DeployModel,
)


class DeployModelFlowParameters:
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        data_gtw: FSDataGateway,
        ion_mode: IonModes,
        n_decimals: int,
        documents_directory: str,
        intensity_weighting_power: Union[float, int] = 0.5,
        allowed_missing_percentage: Union[float, int] = 5.0,
        redis_db: str = "0",
        overwrite_model: bool = False,
    ):
        self.data_gtw = data_gtw
        self.spectrum_dgw = spectrum_dgw
        self.documents_directory = documents_directory
        if ion_mode not in ION_MODES:
            raise ValueError("Ion mode can only be either 'positive' or 'negative'.")
        self.embedding = MakeEmbeddingsParameters(
            ion_mode, n_decimals, intensity_weighting_power, allowed_missing_percentage
        )
        self.deploying = DeployModelParameters(
            redis_db, overwrite_model, model_name=f"spec2vec-{ion_mode}"
        )


def build_deploy_model_flow(
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: DeployModelFlowParameters,
) -> Flow:
    """
    Builds the spec2vec machine learning pipeline. It process data, trains a model, makes
    embeddings, registers the model and deploys it to the API.


    Parameters
    ----------
    flow_name:
        Name of the flow
    flow_config: FlowConfig
        Configuration dataclass passed to prefect.Flow as a dict
    flow_parameters:
        Class containing all flow parameters
    Returns
    -------

    """
    p = flow_parameters

    with Flow(flow_name, **flow_config.kwargs) as deploy_model_flow:
        model_run_id = Parameter("ModelRunID")
        document_paths = ListDocumentPaths(p.documents_directory, p.data_gtw)()
        loaded_model = LoadSpec2VecModel(p.data_gtw)(model_run_id)
        _ = MakeEmbeddings(p.spectrum_dgw, p.data_gtw, p.embedding).map(
            unmapped(loaded_model),
            unmapped(model_run_id),
            document_paths,
        )
        _ = DeployModel(flow_parameters.deploying)(model_run_id)

    return deploy_model_flow
