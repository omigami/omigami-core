from omigami.ms2deepscore.tasks.calculate_tanimoto_score import CalculateTanimotoScore
from prefect import Flow


def test_calculate_tanimoto_score(binned_spectra_stored, tmpdir):
    with Flow("test") as flow:
        res = CalculateTanimotoScore(n_bits=2048)()

    state = flow.run()
    assert state.is_successful()
    assert isinstance(state.result[res].result, dict)
