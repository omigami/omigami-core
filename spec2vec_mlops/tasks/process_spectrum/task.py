from dataclasses import dataclass
from typing import List

from prefect import Task

from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.tasks.process_spectrum.spectrum_processor import SpectrumProcessor
from spec2vec_mlops.tasks.config import merge_configs
from spec2vec_mlops.data_gateway import SpectrumDataGateway, InputDataGateway


class ProcessSpectrum(Task):
    def __init__(
        self,
        download_path: str,
        spectrum_dgw: SpectrumDataGateway,
        input_dgw: InputDataGateway,
        n_decimals: int,
        skip_if_exists: bool = True,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._input_dgw = input_dgw
        self._n_decimals = n_decimals
        self._skip_if_exists = skip_if_exists
        self._processor = SpectrumProcessor()
        self._download_path = download_path
        config = merge_configs(kwargs)
        super().__init__(**config)

    def run(self, spectrum_ids: List[str] = None) -> List[str]:
        self.logger.info(
            f"Processing {len(spectrum_ids)} spectra from the input data {self._download_path}"
        )
        spectrum_data = self._input_dgw.load_spectrum_ids(
            self._download_path, spectrum_ids
        )
        # TODO: refactor to use prefect's checkpoint functionality
        if self._skip_if_exists:
            new_spectrum_ids = self._spectrum_dgw.list_spectra_not_exist(spectrum_ids)
            spectrum_data = [
                sp for sp in spectrum_data if sp.get("spectrum_id") in new_spectrum_ids
            ]
            self.logger.info(
                f"{len(new_spectrum_ids)} out of {len(spectrum_ids)} are new and will"
                f"be processed."
            )

        cleaned_data = [
            self._processor.process_data(spectra_data) for spectra_data in spectrum_data
        ]
        cleaned_data = [spectrum for spectrum in cleaned_data if spectrum]
        spectrum_data = [
            SpectrumDocumentData(spectrum, self._n_decimals)
            for spectrum in cleaned_data
        ]

        self.logger.info(f"Finished processing. saving into spectrum database.")
        self._spectrum_dgw.write_spectrum_documents(spectrum_data)
        return [sp.spectrum_id for sp in spectrum_data]


@dataclass
class ProcessSpectrumParameters:
    spectrum_dgw: SpectrumDataGateway
    input_dgw: InputDataGateway
    n_decimals: int = 2
    # TODO: deprecated parameter. see comments on clean data task
    skip_if_exists: bool = True

    @property
    def kwargs(self):
        return dict(
            spectrum_dgw=self.spectrum_dgw,
            input_dgw=self.input_dgw,
            n_decimals=self.n_decimals,
            skip_if_exists=self.skip_if_exists,
        )
