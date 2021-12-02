from prefect import Flow, unmapped, Parameter

from omigami.config import IonModes, ION_MODES, MLFLOW_SERVER
from omigami.flow_config import FlowConfig
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.ms2deepscore.tasks import (
    CreateSpectrumIDsChunks,
    GetMS2DeepScoreModelPath,
)
from omigami.spectra_matching.ms2deepscore.tasks import MakeEmbeddings
from omigami.spectra_matching.tasks import (
    DeployModelParameters,
    DeployModel,
)


class DeployModelFlowParameters:
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        fs_dgw: MS2DeepScoreFSDataGateway,
        ion_mode: IonModes,
        documents_directory: str,
        redis_db: str = "0",
        model_registry_uri: str = MLFLOW_SERVER,
        overwrite_model: bool = False,
        spectrum_ids_chunk_size: int = 10000,
    ):
        self.fs_dgw = fs_dgw
        self.spectrum_dgw = spectrum_dgw
        self.documents_directory = documents_directory
        self.model_registry_uri = model_registry_uri
        self.spectrum_chunk_size = spectrum_ids_chunk_size

        if ion_mode not in ION_MODES:
            raise ValueError("Ion mode can only be either 'positive' or 'negative'.")
        self.ion_mode = ion_mode

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
    Flow:
        A deploy model prefect flow

    """

    with Flow(flow_name, **flow_config.kwargs) as deploy_model_flow:
        model_run_id = Parameter("ModelRunID")

        spectrum_id_chunks = CreateSpectrumIDsChunks(
            flow_parameters.spectrum_chunk_size, flow_parameters.spectrum_dgw
        )()

        model_path = GetMS2DeepScoreModelPath(flow_parameters.model_registry_uri)(
            model_run_id
        )

        MakeEmbeddings(
            flow_parameters.spectrum_dgw,
            flow_parameters.fs_dgw,
            flow_parameters.ion_mode,
        ).map(
            unmapped(model_path),
            unmapped(model_run_id),
            spectrum_id_chunks,
        )

        _ = DeployModel(flow_parameters.deploying)(model_run_id)

    return deploy_model_flow
