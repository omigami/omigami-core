FROM prefecthq/prefect:latest

COPY . /tmp/spec2vec_mlops
RUN pip install -e /tmp/spec2vec_mlops