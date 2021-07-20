from unittest.mock import Mock

from ms2deepscore import BinnedSpectrum, SpectrumBinner

from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.tasks.bin_spectra import BinSpectra


def test_bin_spectra(cleaned_data_ms2deep_score):
    gtw = Mock(spec=MS2DeepScoreRedisSpectrumDataGateway)
    spectrum_binner = BinSpectra(gtw)

    spectrum_binner, binned_spectra = spectrum_binner._bin_spectra(
        cleaned_data_ms2deep_score,
    )

    assert isinstance(spectrum_binner, SpectrumBinner)
    assert isinstance(binned_spectra[0], BinnedSpectrum)
    assert binned_spectra[1].get("spectrum_id") == cleaned_data_ms2deep_score[1].get(
        "spectrum_id"
    )
    assert binned_spectra[3].get("inchi") == cleaned_data_ms2deep_score[3].get("inchi")
    assert [spectra.get("inchikey") for spectra in binned_spectra] == [
        spectra.get("inchikey") for spectra in cleaned_data_ms2deep_score
    ]
