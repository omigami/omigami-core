from datetime import timedelta, date, datetime
from typing import Tuple

from omigami.config import IonModes, ION_MODES
from omigami.flow_config import FlowConfig
from omigami.gateways.data_gateway import DataGateway
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.helper_classes.siamese_model_trainer import SplitRatio
from omigami.ms2deepscore.tasks import (
    ProcessSpectrumParameters,
    ProcessSpectrum,
    RegisterModel,
    DeployModel,
    DeployModelParameters,
    RegisterModelParameters,
)
from omigami.ms2deepscore.tasks.calculate_tanimoto_score import (
    CalculateTanimotoScoreParameters,
    CalculateTanimotoScore,
)
from omigami.ms2deepscore.tasks.train_model import TrainModelParameters, TrainModel
from omigami.spectrum_cleaner import SpectrumCleaner
from omigami.tasks import (
    DownloadData,
    DownloadParameters,
    ChunkingParameters,
    CreateChunks,
    SaveRawSpectraParameters,
    SaveRawSpectra,
)
from prefect import Flow
from prefect.schedules import Schedule
from prefect.schedules.clocks import IntervalClock


class TrainingFlowParameters:
    def __init__(
        self,
        data_gtw: DataGateway,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        spectrum_cleaner: SpectrumCleaner,
        source_uri: str,
        output_dir: str,
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
        mlflow_output_dir: str,
        mlflow_server: str,
        epochs: int = 50,
        learning_rate: float = 0.001,
        layer_base_dims: Tuple[int] = (600, 500, 400),
        embedding_dim: int = 400,
        dropout_rate: float = 0.2,
        train_ratio: float = 0.9,
        validation_ratio: float = 0.05,
        test_ratio: float = 0.05,
        schedule_task_days: int = 30,
        dataset_name: str = "gnps.json",
        dataset_checkpoint_name: str = "spectrum_ids.pkl",
        environment: str = "dev",
        redis_db: str = "0",
    ):
        self.data_gtw = data_gtw
        self.spectrum_dgw = spectrum_dgw

        if schedule_task_days != None:
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
            source_uri, output_dir, dataset_id, dataset_name, dataset_checkpoint_name
        )

        self.chunking = ChunkingParameters(
            self.downloading.download_path, chunk_size, ion_mode
        )
        self.save_raw_spectra = SaveRawSpectraParameters(
            spectrum_dgw, data_gtw, spectrum_cleaner, overwrite_all_spectra
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
            learning_rate,
            layer_base_dims,
            embedding_dim,
            dropout_rate,
            SplitRatio(train_ratio, validation_ratio, test_ratio),
        )

        self.registering = RegisterModelParameters(
            project_name, mlflow_output_dir, mlflow_server, ion_mode
        )

        self.deploying = DeployModelParameters(
            redis_db, ion_mode, overwrite_model, environment
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
    -------

    """

    with Flow(name=flow_name, **flow_config.kwargs) as training_flow:

        spectrum_ids = DownloadData(
            flow_parameters.data_gtw,
            flow_parameters.downloading,
        )()

        gnps_chunks = CreateChunks(
            flow_parameters.data_gtw,
            flow_parameters.chunking,
        )(spectrum_ids)

        spectrum_ids_chunks = SaveRawSpectra(flow_parameters.save_raw_spectra).map(
            gnps_chunks
        )

        processed_ids = ProcessSpectrum(
            flow_parameters.data_gtw,
            flow_parameters.spectrum_dgw,
            flow_parameters.process_spectrum,
        )(spectrum_ids_chunks)

        scores_output_path = CalculateTanimotoScore(
            flow_parameters.spectrum_dgw, flow_parameters.calculate_tanimoto_score
        )(processed_ids)

        ms2deepscore_model_path = TrainModel(
            flow_parameters.data_gtw,
            flow_parameters.spectrum_dgw,
            flow_parameters.training,
        )(processed_ids, scores_output_path)

        model_registry = RegisterModel(flow_parameters.registering)(
            ms2deepscore_model_path
        )

        if deploy_model:
            DeployModel(flow_parameters.deploying)(model_registry)

    return training_flow
