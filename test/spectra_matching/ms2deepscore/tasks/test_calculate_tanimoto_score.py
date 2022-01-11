import os

from prefect import Flow

from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import MS2DeepScoreFSDataGateway

from omigami.spectra_matching.ms2deepscore.tasks import (
    CalculateTanimotoScore,
    CalculateTanimotoScoreParameters,
)


def test_calculate_tanimoto_score(binned_spectra_to_train_path, tmpdir):
    path = f"{tmpdir}/tanimoto_scores.pkl"
    parameters = CalculateTanimotoScoreParameters(
        binned_spectra_path=binned_spectra_to_train_path, scores_output_path=path, n_bits=2048, decimals=5
    )
    with Flow("test") as flow:
        res = CalculateTanimotoScore(
            MS2DeepScoreFSDataGateway(), parameters
        )([""])

    state = flow.run()
    assert state.is_successful()
    assert os.path.exists(path)
