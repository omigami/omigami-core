FROM drtools/alpine-conda:alpine-3.11_conda-4.7.12
ENV PATH="/opt/conda/bin/:${PATH}"
USER root
RUN mkdir /opt/omigami
RUN chown -R anaconda /opt/omigami
USER anaconda
WORKDIR /opt/omigami

ENV PIP_FIND_LINKS=/opt/libs
COPY ./libs /opt/libs

COPY ./requirements/requirements_flow.txt /opt/omigami/requirements/requirements_flow.txt
COPY ./omigami/spectra_matching/requirements/requirements.txt /opt/omigami/requirements/requirements_spectra_matching.txt

RUN conda install -y -c conda-forge -c nlesc -c bioconda python=3.7 \
--file ./requirements/requirements_flow.txt \
--file ./requirements/requirements_spectra_matching.txt \
&& /opt/conda/bin/conda clean -afy \
&& find /opt/conda/ -follow -type f -name '*.a' -delete \
&& find /opt/conda/ -follow -type f -name '*.pyc' -delete \
&& find /opt/conda/ -follow -type f -name '*.js.map' -delete

COPY ./requirements/requirements_flow_pip.txt /opt/omigami/requirements/requirements_flow_pip.txt
RUN pip install -r requirements/requirements_flow_pip.txt

COPY ./omigami/spectra_matching/requirements/requirements_pip.txt /opt/omigami/requirements/requirements_spectra_matching_pip.txt
RUN pip install -r requirements/requirements_spectra_matching_pip.txt
