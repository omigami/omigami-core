ARG BASE_IMAGE

FROM $BASE_IMAGE

WORKDIR /opt/omigami

ENV PIP_DEFAULT_TIMEOUT=100

COPY ./omigami/spectra_matching/ms2deepscore/requirements/requirements.txt /opt/omigami/requirements/requirements_ms2ds.txt

RUN conda install -y -c conda-forge -c nlesc -c bioconda \
--file ./requirements/requirements_ms2ds.txt \
&& /opt/conda/bin/conda clean -afy \
&& find /opt/conda/ -follow -type f -name '*.a' -delete \
&& find /opt/conda/ -follow -type f -name '*.pyc' -delete \
&& find /opt/conda/ -follow -type f -name '*.js.map' -delete

COPY ./omigami/spectra_matching/ms2deepscore/requirements/requirements_pip.txt /opt/omigami/requirements/requirements_ms2ds_pip.txt
RUN pip install --no-cache-dir -r requirements/requirements_ms2ds_pip.txt

COPY . /opt/omigami
RUN pip install -e /opt/omigami/
