FROM prefecthq/prefect:latest

COPY . /tmp/spec2vec-mlops
RUN pip install -e /tmp/spec2vec-mlops