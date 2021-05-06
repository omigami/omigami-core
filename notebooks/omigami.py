import pandas as pd
import requests
import json

try:
    from matchms.importing import load_from_mgf
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "matchms"])
    from matchms.importing import load_from_mgf


def run(mgf_file: str, n_best_spectra: int = 10):
    # Load the spectra from the MGF file into a generator
    spectra_generator = load_from_mgf(mgf_file)

    payload = build_payload(spectra_generator, n_best_spectra)

    API_request = call_prediction_API(payload)

    predicted_spectra = format_predictions(API_request)

    return predicted_spectra


def build_payload(spectra_generator, n_best_spectra):
    # extract abundance pairs and Precursor_MZ data
    spectra = []
    for spectrum in spectra_generator:
        spectra.append(
            {
                "peaks_json": str(
                    [[mz, intensity] for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities)]),
                "Precursor_MZ": str(spectrum.metadata['pepmass'][0])
            }
        )

    # build the payload
    payload = {
        "data": {
            "ndarray": {
                "parameters":
                    {
                        "n_best_spectra": n_best_spectra,
                    },
                "data": spectra
            }
        }
    }

    return payload


def call_prediction_API(payload):
    # url of the API endpoint
    url = "https://mlops.datarevenue.com/seldon/seldon/spec2vec/api/v0.1/predictions"

    # Query of the prediction API
    api_request = requests.post(url, json=payload)

    return api_request


def format_predictions(api_request):
    # formatting of the results
    response = json.loads(api_request.text)
    predicted_spectra_raw = response['data']['ndarray']

    predicted_spectra = []
    for i in range(len(predicted_spectra_raw)):
        prediction_dataframe = pd.DataFrame(data=[spectrum_id['score'] for spectrum_id in predicted_spectra_raw[i]],
                                            index=[spectrum_id['match_spectrum_id'] for spectrum_id in
                                                   predicted_spectra_raw[i]],
                                            columns=['score'])
        prediction_dataframe.index.name = f'matches of spectrum #{i + 1}'
        predicted_spectra.append(prediction_dataframe)

    return predicted_spectra
