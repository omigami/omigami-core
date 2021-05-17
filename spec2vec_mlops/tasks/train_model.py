import datetime

import gensim
import prefect
from attr import dataclass
from prefect import Task

from spec2vec_mlops.helper_classes.model_trainer import spec2vec_settings
from spec2vec_mlops.gateways.redis_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.tasks.config import merge_configs


@dataclass
class TrainModelParameters:
    """
    iterations:
        Number of training iterations
    window:
        Window size for context around the word
    """

    spectrum_dgw: RedisSpectrumDataGateway
    iterations: int = 25
    window: int = 500


class TrainModel(Task):
    def __init__(self, spectrum_dgw: RedisSpectrumDataGateway, **kwargs):
        self.spectrum_dgw = spectrum_dgw

        config = merge_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, iterations: int = 25, window: int = 500):
        logger = prefect.context.get("logger")
        beg = datetime.datetime.now()

        documents = self.spectrum_dgw.read_documents_iter()
        callbacks, settings = spec2vec_settings(iterations=iterations, window=window)
        model = gensim.models.Word2Vec(
            sentences=documents, callbacks=callbacks, **settings
        )
        logger.info(f"Train model in {datetime.datetime.now() - beg} hours.")

        return model
