import os

from prefect import Flow

from omigami.ms2deepscore.tasks import CalculateTanimotoScore


def test_calculate_tanimoto_score(binned_spectra_stored, tmpdir):
    path = f"{tmpdir}/tanimoto_scores.pkl"
    with Flow("test") as flow:
        res = CalculateTanimotoScore(path, n_bits=2048)()

    state = flow.run()
    assert state.is_successful()
    assert os.path.exists(path)
