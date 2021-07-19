from unittest.mock import Mock

from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.tasks.train_model import TrainModel


def test_train_model(binned_spectra, binned_spectra_tanimoto_score):
    gtw = Mock(spec=MS2DeepScoreRedisSpectrumDataGateway)
    gtw.read_binned_spectra = Mock(return_value=binned_spectra)
    gtw.read_tanimoto_score = Mock(return_value=binned_spectra_tanimoto_score)

    # draft
    gtw.get_spectrum_binner = Mock(return_value=None)

    train_model = TrainModel(spectrum_dgw=gtw)

    train_model.run()
