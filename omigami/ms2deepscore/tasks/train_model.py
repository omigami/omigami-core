from dataclasses import dataclass

from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.utils import merge_prefect_task_configs
from prefect import Task


@dataclass
class TrainModelParameters:
    epochs: int = 50
    learning_rate: float = 0.001


class TrainModel(Task):
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        train_paramaters: TrainModelParameters,
        **kwargs,
    ):
        self._spectrum_gtw = spectrum_dgw
        self._epochs = train_paramaters.epochs
        self._learning_rate = train_paramaters.learning_rate

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self):
        pass
