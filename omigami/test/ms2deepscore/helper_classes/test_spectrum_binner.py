from omigami.ms2deepscore.helper_classes.spectrum_binner import (
    MS2DeepScoreSpectrumBinner,
)


def test_bin_spectra(cleaned_data_ms2deep_score):
    spectrum_binner = MS2DeepScoreSpectrumBinner()
    binned_spectra = spectrum_binner.bin_spectra(
        cleaned_data_ms2deep_score,
    )

    expected_binning = spectrum_binner.spectrum_binner.transform(
        [cleaned_data_ms2deep_score[0]]
    )
    assert binned_spectra[0].binned_peaks == expected_binning[0].binned_peaks
    assert spectrum_binner.spectrum_binner.known_bins is not None
    assert (
        binned_spectra[1].metadata["spectrum_id"]
        == cleaned_data_ms2deep_score[1].metadata["spectrum_id"]
    )
