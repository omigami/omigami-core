source activate omigami
conda env update --name omigami --file requirements/requirements.txt \
    --file requirements/requirements_flow.txt \
    --file requirements/requirements_test.txt \
    --file omigami/spectra_matching/requirements/requirements.txt \
    --file omigami/spectra_matching/spec2vec/requirements/requirements.txt \
    --file omigami/spectra_matching/ms2deepscore/requirements/requirements.txt
pip install -r requirements/requirements_flow_pip.txt \
    -r requirements/requirements_test_pip.txt \
    -r omigami/spectra_matching/requirements/requirements_pip.txt \
    -r omigami/spectra_matching/spec2vec/requirements/requirements_pip.txt \
    -r omigami/spectra_matching/ms2deepscore/requirements/requirements_pip.txt
