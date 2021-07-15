from ms2deepscore import BinnedSpectrum
from omigami.ms2deepscore.helper_classes.spectrum_binner import (
    MS2DeepScoreSpectrumBinner,
)


def test_bin_spectra(cleaned_data_ms2deep_score, ms2deepscore_real_model_path):
    spectrum_binner = MS2DeepScoreSpectrumBinner()
    binned_spectra = spectrum_binner.bin_spectra(
        cleaned_data_ms2deep_score,
    )

    assert isinstance(binned_spectra[0], BinnedSpectrum)
    assert (
        binned_spectra[1].metadata["spectrum_id"]
        == cleaned_data_ms2deep_score[1].metadata["spectrum_id"]
    )
