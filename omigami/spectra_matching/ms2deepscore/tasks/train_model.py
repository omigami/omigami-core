from dataclasses import dataclass
from typing import Dict

import prefect
from prefect import Task

from omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer import (
    SiameseModelTrainer,
    SplitRatio,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.utils import merge_prefect_task_configs


@dataclass
class TrainModelParameters:
    output_path: str
    spectrum_binner_output_path: str
    binned_spectra_path: str
    epochs: int = 50
    split_ratio: SplitRatio = SplitRatio()


class TrainModel(Task):
    def __init__(
        self,
        fs_dgw: MS2DeepScoreFSDataGateway,
        train_parameters: TrainModelParameters,
        **kwargs,
    ):
        self._fs_dgw = fs_dgw
        self._spectrum_binner_output_path = train_parameters.spectrum_binner_output_path
        self._binned_spectra_path = train_parameters.binned_spectra_path
        self._output_path = train_parameters.output_path
        self._epochs = train_parameters.epochs
        self._split_ratio = train_parameters.split_ratio

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(
        self,
        scores_output_path: str = None,
    ) -> Dict:
        """
        Prefect task to train SiameseModel on given spectra.

        Parameters
        ----------
        scores_output_path: str
            Output path to save resulting similarity scores

        Returns
        -------
        Dictionary containing `ms2deepscore_model_path` and `validation_loss`

        """
        spectrum_binner = self._fs_dgw.read_from_file(self._spectrum_binner_output_path)

        trainer = SiameseModelTrainer(
            self._fs_dgw,
            self._binned_spectra_path,
            self._epochs,
            self._split_ratio,
        )
        model = trainer.train(scores_output_path, spectrum_binner, self.logger)

        output_path = self._output_path.format(
            flow_run_id=prefect.context.get("flow_run_id", "local")
        )
        self.logger.info(f"Saving trained model to {output_path}.")
        self._fs_dgw.save(model, output_path)

        return {
            "ms2deepscore_model_path": output_path,
            "validation_loss": model.model.history.history["val_loss"][-1],
        }
