ENV PIP_DEFAULT_TIMEOUT=100

COPY ./omigami/spectra_matching/ms2deepscore/requirements /opt/ms2deepscore/requirements
RUN cat requirements/environment.frozen.yaml | sed 's/omigami/base/g' > environment-docker.yml

RUN /opt/conda/bin/conda env update --file environment-docker.yml \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete

COPY . /opt/ms2deepscore
