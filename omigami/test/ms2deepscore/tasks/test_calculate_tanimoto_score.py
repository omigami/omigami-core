import os

from prefect import Flow

from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.tasks import CalculateTanimotoScore
from omigami.ms2deepscore.tasks.calculate_tanimoto_score import (
    CalculateTanimotoScoreParameters,
)


def test_calculate_tanimoto_score(binned_spectra_stored, binned_spectra, tmpdir):
    spectrum_ids = [spectrum.get("spectrum_id") for spectrum in binned_spectra]
    path = f"{tmpdir}/tanimoto_scores.pkl"
    parameters = CalculateTanimotoScoreParameters(
        scores_output_path=path, ion_mode="positive", n_bits=2048, decimals=5
    )
    with Flow("test") as flow:
        res = CalculateTanimotoScore(
            MS2DeepScoreRedisSpectrumDataGateway(), parameters
        )(spectrum_ids)

    state = flow.run()
    assert state.is_successful()
    assert os.path.exists(path)
