FROM drtools/alpine-conda:alpine-3.11_conda-4.7.12
ENV PATH="/opt/conda/bin/:${PATH}"
USER root
RUN mkdir /opt/omigami
RUN chown -R anaconda /opt/omigami
USER anaconda
WORKDIR /opt/omigami

ENV PIP_FIND_LINKS=/opt/libs
COPY ./libs /opt/libs

COPY ./requirements/spectra_matching/conda.txt /opt/omigami/requirements/conda.txt

RUN conda install -y -c conda-forge -c nlesc -c bioconda python=3.7 \
--file ./requirements/conda.txt \
&& /opt/conda/bin/conda clean -afy \
&& find /opt/conda/ -follow -type f -name '*.a' -delete \
&& find /opt/conda/ -follow -type f -name '*.pyc' -delete \
&& find /opt/conda/ -follow -type f -name '*.js.map' -delete

COPY ./requirements/spectra_matching/pip.txt /opt/omigami/requirements/pip.txt
RUN pip install -r requirements/pip.txt
