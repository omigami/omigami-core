FROM continuumio/miniconda3
ENV PATH="/opt/conda/bin/:${PATH}"
WORKDIR /opt/spec2vec_mlops

COPY . /opt/spec2vec_mlops
RUN cat requirements/environment.frozen.yaml | sed 's/spec2vec_mlops/base/g' > environment-docker.yml

RUN /opt/conda/bin/conda env update --file environment-docker.yml

RUN pip install -e /opt/spec2vec_mlops
