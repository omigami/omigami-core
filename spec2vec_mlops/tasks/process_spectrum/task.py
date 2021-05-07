from dataclasses import dataclass
from typing import List

from prefect import Task

from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.tasks.process_spectrum.spectrum_processor import SpectrumProcessor
from spec2vec_mlops.tasks.config import merge_configs
from spec2vec_mlops.tasks.data_gateway import SpectrumDataGateway, InputDataGateway


class ProcessSpectrum(Task):
    def __init__(
        self,
        spectrum_dgw: SpectrumDataGateway,
        input_dgw: InputDataGateway,
        n_decimals: int,
        chunk_size: int,
        skip_if_exists: bool = True,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._input_dgw = input_dgw
        self._n_decimals = n_decimals
        self._chunk_size = chunk_size
        self._skip_if_exists = skip_if_exists
        self._processor = SpectrumProcessor()
        config = merge_configs(kwargs)
        super().__init__(**config)

    def run(self, spectrum_data: List[dict] = None) -> List[str]:
        # TODO: refactor to use prefect's checkpoint functionality
        if self._skip_if_exists:
            spectrum_ids = [sp.get("spectrum_id") for sp in spectrum_data]
            spectrum_ids = self._spectrum_dgw.list_spectra_not_exist(spectrum_ids)
            spectrum_data = [
                sp for sp in spectrum_data if sp.get("spectrum_id") in spectrum_ids
            ]

        cleaned_data = [
            self._processor.process_data(spectra_data) for spectra_data in spectrum_data
        ]
        cleaned_data = [spectrum for spectrum in cleaned_data if spectrum]
        spectrum_data = [
            SpectrumDocumentData(spectrum, self._n_decimals)
            for spectrum in cleaned_data
        ]

        self._spectrum_dgw.write_spectrum_documents(spectrum_data)
        return [sp.spectrum_id for sp in spectrum_data]


@dataclass
class ProcessSpectrumParameters:
    spectrum_dgw: SpectrumDataGateway
    n_decimals: int = 2
    chunk_size: int = 2000
    # TODO: deprecated parameter. see comments on clean data task
    skip_if_exists: bool = True

    @property
    def kwargs(self):
        return dict(
            spectrum_dgw=self.spectrum_dgw,
            n_decimals=self.n_decimals,
            skip_if_exists=self.skip_if_exists,
        )
