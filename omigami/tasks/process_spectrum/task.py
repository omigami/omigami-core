from dataclasses import dataclass
from typing import List

from prefect import Task

from omigami.tasks.process_spectrum.spectrum_processor import SpectrumProcessor
from omigami.tasks.config import merge_configs
from omigami.data_gateway import SpectrumDataGateway, InputDataGateway

from omigami.gateways.redis_spectrum_gateway import REDIS_DB


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

    def run(self, gnps_path: str = None) -> List[str]:
        self.logger.debug(f"Using Redis DB {REDIS_DB}")
        spectra = self._input_dgw.load_spectrum(gnps_path)
        self.logger.info(
            f"Processing {len(spectra)} spectra from the input data {gnps_path}"
        )
        spectrum_ids = [sp["SpectrumID"] for sp in spectra]

        # TODO: refactor to use prefect's checkpoint functionality
        self.logger.info(f"Flag skip_if_exists is set to {self._skip_if_exists}.")
        if self._skip_if_exists:
            new_spectrum_ids = self._spectrum_dgw.list_spectra_not_exist(spectrum_ids)
            if not new_spectrum_ids:
                self.logger.info("All spectra have already been processed.")
                return spectrum_ids

            self.logger.info(
                f"{len(new_spectrum_ids)} out of {len(spectrum_ids)} spectra are new and will "
                f"be processed."
            )
            spectra = [sp for sp in spectra if sp["SpectrumID"] in new_spectrum_ids]

        self.logger.info(f"Processing spectra and converting into documents.")
        spectrum_documents = self._processor.create_documents(
            spectra, n_decimals=self._n_decimals
        )

        self.logger.info(f"Finished processing. saving into spectrum database.")
        self._spectrum_dgw.write_spectrum_documents(spectrum_documents)

        return [sp.spectrum_id for sp in spectrum_documents]


@dataclass
class ProcessSpectrumParameters:
    spectrum_dgw: SpectrumDataGateway
    n_decimals: int = 2
    # TODO: deprecated parameter. see comments on clean data task
    skip_if_exists: bool = True

    @property
    def kwargs(self):
        return dict(
            spectrum_dgw=self.spectrum_dgw,
            n_decimals=self.n_decimals,
            skip_if_exists=self.skip_if_exists,
        )
