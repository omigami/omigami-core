FROM continuumio/miniconda3
ENV PATH="/opt/conda/bin/:${PATH}"
WORKDIR /opt/omigami

COPY ./requirements /opt/omigami/requirements
RUN cat requirements/environment.frozen.yaml | sed 's/omigami/base/g' > environment-docker.yml

RUN /opt/conda/bin/conda env update --file environment-docker.yml \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete

COPY . /opt/omigami
RUN pip install -e /opt/omigami
