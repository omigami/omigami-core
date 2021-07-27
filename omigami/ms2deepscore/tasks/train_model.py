from dataclasses import dataclass
from typing import Tuple, List

from omigami.gateways import DataGateway
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.helper_classes.siamese_model_trainer import (
    SiameseModelTrainer,
    SplitRatio,
)
from omigami.utils import merge_prefect_task_configs
from prefect import Task


@dataclass
class TrainModelParameters:
    output_path: str
    spectrum_binner_output_path: str
    epochs: int = 50
    learning_rate: float = 0.001
    layer_base_dims: Tuple[int] = (600, 500, 400)
    embedding_dim: int = 400
    dropout_rate: float = 0.2
    split_ratio: SplitRatio = SplitRatio()


class TrainModel(Task):
    def __init__(
        self,
        fs_gtw: DataGateway,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        train_parameters: TrainModelParameters,
        **kwargs,
    ):
        self._fs_gtw = fs_gtw
        self._spectrum_gtw = spectrum_dgw
        self._spectrum_binner_output_path = train_parameters.spectrum_binner_output_path
        self._output_path = train_parameters.output_path
        self._epochs = train_parameters.epochs
        self._learning_rate = train_parameters.learning_rate
        self._layer_base_dims = train_parameters.layer_base_dims
        self._embedding_dim = train_parameters.embedding_dim
        self._dropout_rate = train_parameters.dropout_rate
        self._split_ratio = train_parameters.split_ratio

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        spectrum_ids: List[str] = None,
        scores_output_path: str = None,
    ) -> str:
        spectrum_binner = self._fs_gtw.read_from_file(self._spectrum_binner_output_path)

        trainer = SiameseModelTrainer(
            self._spectrum_gtw,
            self._epochs,
            self._learning_rate,
            self._layer_base_dims,
            self._embedding_dim,
            self._dropout_rate,
            self._split_ratio,
        )
        model = trainer.train(
            spectrum_ids, scores_output_path, spectrum_binner, self.logger
        )
        self._fs_gtw.save(model, self._output_path)
        return self._output_path
