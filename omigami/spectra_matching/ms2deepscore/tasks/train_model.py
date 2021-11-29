from dataclasses import dataclass
from typing import List, Dict

from prefect import Task

from omigami.config import IonModes
from omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer import (
    SiameseModelTrainer,
    SplitRatio,
)
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.storage import DataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class TrainModelParameters:
    output_path: str
    ion_mode: IonModes
    spectrum_binner_output_path: str
    epochs: int = 50
    split_ratio: SplitRatio = SplitRatio()


class TrainModel(Task):
    def __init__(
        self,
        fs_gtw: DataGateway,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        train_parameters: TrainModelParameters,
        **kwargs,
    ):
        self._fs_dgw = fs_gtw
        self._spectrum_gtw = spectrum_dgw
        self._ion_mode = train_parameters.ion_mode
        self._spectrum_binner_output_path = train_parameters.spectrum_binner_output_path
        self._output_path = train_parameters.output_path
        self._epochs = train_parameters.epochs
        self._split_ratio = train_parameters.split_ratio

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        spectrum_ids: List[str] = None,
        scores_output_path: str = None,
    ) -> Dict:
        """
        Prefect task to train SiameseModel on given spectra.

        Parameters
        ----------
        spectrum_ids: List[str]
            spectrum_ids to train model on
        scores_output_path: str
            Output path to save resulting similarity scores

        Returns
        -------
        Dictionary containing `ms2deepscore_model_path` and `validation_loss`

        """
        spectrum_binner = self._fs_dgw.read_from_file(self._spectrum_binner_output_path)

        trainer = SiameseModelTrainer(
            self._spectrum_gtw,
            self._ion_mode,
            self._epochs,
            self._split_ratio,
        )
        model = trainer.train(
            spectrum_ids, scores_output_path, spectrum_binner, self.logger
        )
        self.logger.info(f"Saving trained model to {self._output_path}.")
        self._fs_dgw.save(model, self._output_path)

        return {
            "ms2deepscore_model_path": self._output_path,
            "validation_loss": model.model.history.history["val_loss"][-1],
        }
