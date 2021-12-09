from datetime import timedelta, date, datetime
from typing import Optional

from prefect import Flow, unmapped
from prefect.schedules import Schedule
from prefect.schedules.clocks import IntervalClock

from omigami.config import IonModes, ION_MODES, MLFLOW_SERVER
from omigami.flow_config import FlowConfig
from omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer import (
    SplitRatio,
)
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.ms2deepscore.tasks import (
    ProcessSpectrumParameters,
    ProcessSpectrum,
    RegisterModel,
    RegisterModelParameters,
    MakeEmbeddings,
    CreateSpectrumIDsChunks,
    CalculateTanimotoScoreParameters,
    CalculateTanimotoScore,
    TrainModelParameters,
    TrainModel,
)
from omigami.spectra_matching.tasks import (
    DownloadParameters,
    DownloadData,
    ChunkingParameters,
    CreateChunks,
    CleanRawSpectraParameters,
    CleanRawSpectra,
    DeployModelParameters,
    DeployModel,
)


class TrainingFlowParameters:
    def __init__(
        self,
        data_gtw: MS2DeepScoreFSDataGateway,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        source_uri: str,
        dataset_directory: str,
        dataset_id: str,
        chunk_size: int,
        ion_mode: IonModes,
        scores_output_path: str,
        fingerprint_n_bits: int,
        scores_decimals: int,
        overwrite_all_spectra: bool,
        spectrum_binner_output_path: str,
        spectrum_binner_n_bins: int,
        overwrite_model: bool,
        model_output_path: str,
        project_name: str,
        mlflow_output_directory: str,
        model_registry_uri: str = MLFLOW_SERVER,
        epochs: int = 50,
        train_ratio: float = 0.9,
        validation_ratio: float = 0.05,
        test_ratio: float = 0.05,
        spectrum_ids_chunk_size: int = 10000,
        schedule_task_days: Optional[int] = 30,
        dataset_name: str = "gnps.json",
        redis_db: str = "0",
    ):
        self.data_gtw = data_gtw
        self.spectrum_dgw = spectrum_dgw
        self.spectrum_chunk_size = spectrum_ids_chunk_size
        self.ion_mode = ion_mode

        if schedule_task_days is not None:
            self.schedule = Schedule(
                clocks=[
                    IntervalClock(
                        start_date=datetime.combine(date.today(), datetime.min.time()),
                        interval=timedelta(days=schedule_task_days),
                    )
                ]
            )
        else:
            self.schedule = None

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

        self.process_spectrum = ProcessSpectrumParameters(
            spectrum_binner_output_path,
            ion_mode=ion_mode,
            n_bins=spectrum_binner_n_bins,
        )

        self.calculate_tanimoto_score = CalculateTanimotoScoreParameters(
            scores_output_path, ion_mode, fingerprint_n_bits, scores_decimals
        )

        self.training = TrainModelParameters(
            model_output_path,
            ion_mode,
            spectrum_binner_output_path,
            epochs,
            SplitRatio(train_ratio, validation_ratio, test_ratio),
        )

        self.registering = RegisterModelParameters(
            project_name, model_registry_uri, mlflow_output_directory, ion_mode
        )

        self.deploying = DeployModelParameters(
            redis_db, overwrite_model, f"ms2deepscore-{ion_mode}"
        )


def build_training_flow(
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: TrainingFlowParameters,
    deploy_model: bool = False,
) -> Flow:
    """
    Builds the MS2DeepScore machine learning pipeline. It Downloads and process data, trains the model, makes
    embeddings, registers the model and deploys it to the API.


    Parameters
    ----------
    flow_name:
        Name of the flow
    flow_config: FlowConfig
        Configuration dataclass passed to prefect.Flow as a dict
    flow_parameters:
        Dataclass containing all flow parameters
    deploy_model:
        If the model will be deployed or not
    -------

    """

    with Flow(name=flow_name, **flow_config.kwargs) as training_flow:
        # tasks return objects, so that we could connect tasks using returned objects.
        # if you update one return type of a task please mind the later tasks too

        spectrum_ids = DownloadData(
            flow_parameters.data_gtw,
            flow_parameters.downloading,
        )()

        gnps_chunk_paths = CreateChunks(
            flow_parameters.data_gtw,
            flow_parameters.chunking,
        )(spectrum_ids)

        cleaned_paths = CleanRawSpectra(
            flow_parameters.data_gtw, flow_parameters.clean_raw_spectra
        ).map(gnps_chunk_paths)

        processed_ids = ProcessSpectrum(
            flow_parameters.data_gtw,
            flow_parameters.spectrum_dgw,
            flow_parameters.process_spectrum,
        )(cleaned_paths)

        scores_output_path = CalculateTanimotoScore(
            flow_parameters.spectrum_dgw, flow_parameters.calculate_tanimoto_score
        )(processed_ids)

        train_model_output = TrainModel(
            flow_parameters.data_gtw,
            flow_parameters.spectrum_dgw,
            flow_parameters.training,
            nout=2,
        )(processed_ids, scores_output_path)

        run_id = RegisterModel(
            flow_parameters.registering, training_parameters=flow_parameters.training
        )(train_model_output)

        processed_chunks = CreateSpectrumIDsChunks(
            flow_parameters.spectrum_chunk_size, flow_parameters.spectrum_dgw
        )(processed_ids)

        MakeEmbeddings(
            flow_parameters.spectrum_dgw,
            flow_parameters.data_gtw,
            flow_parameters.ion_mode,
        ).map(
            unmapped(train_model_output),
            unmapped(run_id),
            processed_chunks,
        )

        if deploy_model:
            DeployModel(flow_parameters.deploying)(run_id)

    return training_flow
