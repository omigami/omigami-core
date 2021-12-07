FROM drtools/alpine-conda:alpine-3.11_conda-4.7.12
ENV PATH="/opt/conda/bin/:${PATH}"
WORKDIR /opt/omigami

ENV PIP_FIND_LINKS=/opt/libs
COPY ./libs /opt/libs

COPY ./requirements/spectra_matching/requirements_conda.txt /opt/omigami/requirements/requirements_conda.txt

RUN conda install -y -c conda-forge -c nlesc -c bioconda python=3.7 \
--file ./requirements/requirements_conda.txt \
&& /opt/conda/bin/conda clean -afy \
&& find /opt/conda/ -follow -type f -name '*.a' -delete \
&& find /opt/conda/ -follow -type f -name '*.pyc' -delete \
&& find /opt/conda/ -follow -type f -name '*.js.map' -delete

COPY ./requirements/spectra_matching/requirements_pip.txt /opt/omigami/requirements/requirements_pip.txt
RUN pip install -r requirements/requirements_pip.txt

COPY . /opt/omigami
RUN pip install /opt/omigami
