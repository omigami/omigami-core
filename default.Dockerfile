FROM continuumio/miniconda3
ENV PATH="/opt/conda/bin/:${PATH}"
WORKDIR /opt/spec2vec_mlops

COPY ./requirements /opt/spec2vec_mlops/requirements
RUN cat requirements/environment.frozen.yaml | sed 's/spec2vec_mlops/base/g' > environment-docker.yml

RUN /opt/conda/bin/conda env update --file environment-docker.yml \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete

RUN apt-get install -y curl
RUN curl "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.0/hadoop-aws-3.3.0.jar" \
    > "/opt/conda/lib/python3.7/site-packages/pyspark/jars/hadoop-aws-3.3.0.jar"
RUN curl "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.11.993/aws-java-sdk-1.11.993.jar" \
    > "/opt/conda/lib/python3.7/site-packages/pyspark/jars/aws-java-sdk-1.11.993.jar"

COPY . /opt/spec2vec_mlops
RUN pip install -e /opt/spec2vec_mlops
