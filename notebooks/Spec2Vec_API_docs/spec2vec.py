import pandas as pd
import requests
import json
from matchms.importing import load_from_mgf


def run(mgf_file: str, n_best_spectra: int = 10):
    """Load the spectra from the MGF file into a generator, build the payload,
    call the prediction API and then format the predictions"""
    spectra_generator = load_from_mgf(mgf_file)

    payload = build_payload(spectra_generator, n_best_spectra)

    API_request = call_spec2vec_API(payload)

    library_spectra = format_results(API_request)

    return library_spectra


def build_payload(spectra_generator, n_best_spectra):
    """Extract abundance pairs and Precursor_MZ data, then build the json payload"""
    spectra = []
    for spectrum in spectra_generator:
        spectra.append(
            {
                "peaks_json": str(
                    [
                        [mz, intensity]
                        for mz, intensity in zip(
                            spectrum.peaks.mz, spectrum.peaks.intensities
                        )
                    ]
                ),
                "Precursor_MZ": str(spectrum.metadata["pepmass"][0]),
            }
        )

    # build the payload
    payload = {
        "data": {
            "ndarray": {
                "parameters": {
                    "n_best_spectra": n_best_spectra,
                },
                "data": spectra,
            }
        }
    }

    return payload


def call_spec2vec_API(payload):
    """"Query of the prediction API endpoint"""
    url = "https://mlops.datarevenue.com/seldon/seldon/spec2vec/api/v0.1/predictions"
    api_request = requests.post(url, json=payload, timeout=600)

    return api_request


def format_results(api_request):
    """Formatting of the results"""
    response = json.loads(api_request.text)
    library_spectra_raw = response["data"]["ndarray"]

    library_spectra_matches = []
    for i in range(len(library_spectra_raw)):
        library_spectra_dataframe = pd.DataFrame(
            data=[spectrum_id["score"] for spectrum_id in library_spectra_raw[i]],
            index=[
                spectrum_id["match_spectrum_id"]
                for spectrum_id in library_spectra_raw[i]
            ],
            columns=["score"],
        )
        library_spectra_dataframe.index.name = f"matches of spectrum #{i + 1}"
        library_spectra_matches.append(library_spectra_dataframe)

    return library_spectra_matches
