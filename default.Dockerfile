FROM continuumio/miniconda3
ENV PATH="/opt/conda/bin/:${PATH}"
WORKDIR /opt/spec2vec_mlops

COPY . /opt/spec2vec_mlops/requirements
RUN cat requirements/environment.frozen.yaml | sed 's/spec2vec_mlops/base/g' > environment-docker.yml

RUN /opt/conda/bin/conda env update --file environment-docker.yml \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete

COPY . /opt/spec2vec_mlops
RUN pip install -e /opt/spec2vec_mlops
