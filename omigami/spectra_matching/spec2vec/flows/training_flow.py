from typing import Union, Optional

from prefect import Flow, unmapped

from omigami.config import IonModes, ION_MODES, MLFLOW_SERVER
from omigami.flow_config import FlowConfig
from omigami.spectra_matching.spec2vec.tasks import (
    CreateDocumentsParameters,
    DeleteEmbeddings,
)
from omigami.spectra_matching.spec2vec.tasks import (
    MakeEmbeddings,
    CreateDocuments,
    TrainModel,
    TrainModelParameters,
    RegisterModel,
    RegisterModelParameters,
    MakeEmbeddingsParameters,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway, FSDataGateway
from omigami.spectra_matching.tasks import (
    DownloadData,
    DownloadParameters,
    ChunkingParameters,
    CreateChunks,
    CleanRawSpectra,
    CleanRawSpectraParameters,
    DeployModelParameters,
    DeployModel,
    CacheCleanedSpectra,
)


class TrainingFlowParameters:
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        fs_dgw: FSDataGateway,
        source_uri: str,
        dataset_directory: str,
        chunk_size: int,
        ion_mode: IonModes,
        n_decimals: int,
        iterations: int,
        window: int,
        mlflow_output_directory: str,
        documents_save_directory: str,
        model_registry_uri: str = MLFLOW_SERVER,
        intensity_weighting_power: Union[float, int] = 0.5,
        allowed_missing_percentage: Union[float, int] = 5.0,
        dataset_name: str = "gnps.json",
        redis_db: str = "0",
        overwrite_model: bool = False,
        model_name: Optional[str] = "spec2vec-model",
        experiment_name: str = "default",
    ):
        self.fs_dgw = fs_dgw
        self.spectrum_dgw = spectrum_dgw
        self.ion_mode = ion_mode
        if ion_mode not in ION_MODES:
            raise ValueError("Ion mode can only be either 'positive' or 'negative'.")

        self.downloading = DownloadParameters(
            source_uri=source_uri,
            output_directory=dataset_directory,
            file_name=dataset_name,
        )
        self.chunking = ChunkingParameters(
            input_file=self.downloading.download_path,
            output_directory=f"{dataset_directory}/raw/{ion_mode}",
            chunk_size=chunk_size,
            ion_mode=ion_mode,
        )
        self.clean_raw_spectra = CleanRawSpectraParameters(
            output_directory=f"{dataset_directory}/cleaned/{ion_mode}"
        )
        self.create_documents = CreateDocumentsParameters(
            output_directory=documents_save_directory,
            ion_mode=ion_mode,
            n_decimals=n_decimals,
        )
        self.training = TrainModelParameters(iterations, window)
        self.registering = RegisterModelParameters(
            experiment_name=experiment_name,
            model_registry_uri=model_registry_uri,
            mlflow_output_directory=mlflow_output_directory,
            n_decimals=n_decimals,
            ion_mode=ion_mode,
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
            model_name=model_name,
        )
        self.embedding = MakeEmbeddingsParameters(
            ion_mode, n_decimals, intensity_weighting_power, allowed_missing_percentage
        )
        self.deploying = DeployModelParameters(
            redis_db, overwrite_model, model_name=f"spec2vec-{ion_mode}"
        )


def build_training_flow(
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: TrainingFlowParameters,
    deploy_model: bool = False,
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
    deploy_model:
        Whether to create a seldon deployment with the result of the training flow.
    Returns
    -------

    """
    with Flow(flow_name, **flow_config.kwargs) as training_flow:
        # tasks return objects, so that we could connect tasks using returned objects.
        # if you update one return type of a task please mind the later tasks too

        spectrum_ids = DownloadData(
            flow_parameters.fs_dgw,
            flow_parameters.downloading,
        )()

        raw_spectra_paths = CreateChunks(
            flow_parameters.fs_dgw,
            flow_parameters.chunking,
        )(spectrum_ids)

        cleaned_spectra_paths = CleanRawSpectra(
            flow_parameters.fs_dgw, flow_parameters.clean_raw_spectra
        ).map(raw_spectra_paths)

        _ = CacheCleanedSpectra(
            flow_parameters.spectrum_dgw, flow_parameters.fs_dgw
        ).map(cleaned_spectra_paths)

        document_paths = CreateDocuments(
            flow_parameters.fs_dgw,
            flow_parameters.create_documents,
        ).map(cleaned_spectra_paths)

        model = TrainModel(flow_parameters.fs_dgw, flow_parameters.training)(
            document_paths
        )

        model_run_id = RegisterModel(flow_parameters.registering)(model)

        if deploy_model:
            delete_embeddings = DeleteEmbeddings(
                flow_parameters.spectrum_dgw, flow_parameters.ion_mode
            )()
            make_embeddings = MakeEmbeddings(
                flow_parameters.spectrum_dgw,
                flow_parameters.fs_dgw,
                flow_parameters.embedding,
            ).map(
                unmapped(model),
                unmapped(model_run_id),
                document_paths,
            )
            make_embeddings.set_dependencies(training_flow, [delete_embeddings])

            deploy_model = DeployModel(flow_parameters.deploying)(model_run_id)
            deploy_model.set_dependencies(training_flow, [make_embeddings])

    return training_flow
