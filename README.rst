##############################
Omigami on MLOps architecture
##############################

Development Environment
=======================

How to setup
------------
::

    export PIP_FIND_LINKS=$(pwd)/libs
    conda create -y -c conda-forge -c nlesc -c bioconda -n omigami python=3.7 \
        --file requirements/requirements.txt \
        --file requirements/requirements_flow.txt \
        --file requirements/requirements_test.txt \
        --file omigami/spectra_matching/requirements/requirements.txt \
        --file omigami/spectra_matching/spec2vec/requirements/requirements.txt \
        --file omigami/spectra_matching/ms2deepscore/requirements/requirements.txt
    source activate omigami
    pip install -r requirements/requirements_flow_pip.txt
    pip install -r requirements/requirements_test_pip.txt
    pip install -r omigami/spectra_matching/requirements/requirements_pip.txt
    pip install -r omigami/spectra_matching/spec2vec/requirements/requirements_pip.txt
    pip install -r omigami/spectra_matching/ms2deepscore/requirements/requirements_pip.txt
    pip install -e .

How to update all packages
--------------------------
To update all packages and create new frozen environments. Make sure you have correct
environment activated. You'll need at least `conda>=4.9`::

    conda activate omigami
    conda env update -f requirements/requirements.txt
    conda env update -f requirements/requirements_flow.txt
    conda env update -f requirements/requirements_test.txt
    conda env update -f omigami/spectra_matching/requirements/requirements.txt
    conda env update -f omigami/spectra_matching/spec2vec/requirements/requirements.txt
    conda env update -f omigami/spectra_matching/ms2deepscore/requirements/requirements.txt
    pip install -r requirements/requirements_flow_pip.txt
    pip install -r requirements/requirements_test_pip.txt
    pip install -r omigami/spectra_matching/requirements/requirements_pip.txt
    pip install -r omigami/spectra_matching/spec2vec/requirements/requirements_pip.txt
    pip install -r omigami/spectra_matching/ms2deepscore/requirements/requirements_pip.txt

How to add or update a single package
-------------------------------------

1. Install the package as usual with conda
2. Check which version was installed and add major and minor version to environment.frozen.yaml
3. Add the package with the most relaxed version restrictions possible to environment.yaml

Special steps for M1 Users
-------------------------------------

1. Activate the conda env and uninstall tensorflow through `pip uninstall tensorflow`
2. Go to https://drive.google.com/drive/folders/11cACNiynhi45br1aW3ub5oQ6SrDEQx4p (source: https://github.com/tensorflow/tensorflow/issues/46044#issuecomment-797151218) and download the `tensorflow-2.5.0-py3-none-any.whl` file.
3. Now do `pip install <path_to_downloaded_file>`
4. You might need to install manually one or two packages. Run the tests and if necessary (you are getting ModuleNotFound errors) install the missing packages.

How to build a docker image
-------------------------------------
Whenever a Flow is registered in Prefect Server using Kubernetes, it needs to use a
Docker image that has Prefect installed and all the packages needed for running the code.

Usually, updating the requirements file should cover all the needed packages. Otherwise,
you should update the Dockerfile.

In order to publish this Docker Image, there's an auxiliary script to do this.
To run it, execute::

    bash deploy.sh

How to run tests that require Redis locally
-------------------------------------------

Some tests that use feature store requires Redis to run.
Start a Redis container and set these environment variables before running the test suite:
::

    docker run -d --rm --name redis -p 6379:6379 redis:5-alpine
    export SKIP_REDIS_TEST=False


Running Prefect Locally
------------------------------------

Start up prefect server.
::
    prefect server start -d


If you are in a M1 machine you might (probably) need to increase docker memory resources to 7 GB.
Alternatively, you can run it through docker-compose (requires less memory):
::
        docker-compose -f local-deployment/docker-compose.yml up -d

To access the dashboard, go to http://localhost:8080/. If you see a blank screen,
you will need to create a tenant:
::
    prefect backend server
    prefect server create-tenant -n default


In a terminal, start an agent that will execute the flows:
::
    prefect agent start -l "dev" --show-flow-logs


To shut down prefect:
::
    prefect server stop
    docker-compose -f local-deployment/docker-compose.yml down  # if you used docker-compose


Running MLFlow Locally
-----------------------

To run mlflow locally run the following command:
::
    mlflow ui --backend-store-uri sqlite:///mlflow.sqlite


To access it: http://localhost:5000/


To run tests one by one via PyCharm, you can add this to your pytest Environment Variables (Run > Edit Configurations...)
::

    SKIP_REDIS_TEST=False;PREFECT__FLOWS__CHECKPOINTING=True;REDIS_HOST=localhost;REDIS_DB=0;MLFLOW_SERVER=sqlite:///mlflow.sqlite

Please don't commit `*model.pkl` files to git. Every necessary model for the
test setup is going to be generated and saved to `test/assets/` folder and be
used from there on.

How to register the training flow manually
------------------------------------------

To register a flow manually to Prefect you need to follow these steps:
::

    conda activate omigami
    export AWS_PROFILE=<your data revenue profile>
    export PYTHONPATH=$(pwd)
    prefect backend server

For Spec2Vec:
::

    pytest omigami/test/spec2vec/test_deployment.py

For MS2DeepScore:
::

    pytest omigami/test/ms2deepscore/test_ms2deepscore_deployment.py

If you want to run the deployment tests in PyCharm,
make sure you have the `AWS_PROFILE` environment variable set in your test configuration
and that you set the Prefect backend to server.

If the Prefect Server requires authentication, you can use the arguments to set it up:
::

    --auth (bool): Enables authentication, defaults to False
    --auth_url (str): Authentication API Path. Ex.: https://mlops.datarevenue.com/.ory/kratos/public/ [Optional, only required if auth=True]
    --username (str): Your username [Optional, only required if auth=True]
    --password (str): Your password [Optional, only required if auth=True]

Then you can check the flow here: https://prefect.mlops.datarevenue.com/default

After the model has been deployed you can access the predictions endpoint in two ways:

By making a curl request:
::

    curl -v https://mlops.datarevenue.com/seldon/seldon/<endpoint-name>/api/v0.1/predictions -H "Content-Type: application/json" -d 'input_data'

::

    curl -v https://mlops.datarevenue.com/seldon/seldon/<endpoint-name>/api/v0.1/predictions -H "Content-Type: application/json" -d @path_to/input.json

By accessing the external API with the user interface at:
::

    https://mlops.datarevenue.com/seldon/seldon/<endpoint-name>/api/v0.1/doc/

Or by querying the prediction API via the python request library (see notebook)


The input data should look like:
::

    {
       "data": {
          "ndarray": {
             "parameters":
                 {
                     "n_best_spectra": 10,
                     "include_metadata": ["Compound_name"]
                 },
             "data":
                 [
                     {"peaks_json": "[[289.286377,8068.000000],[295.545288,22507.000000]]",
                      "Precursor_MZ": "900"},
                     {"peaks_json": "[[289.286377,8068.000000],[295.545288,22507.000000]]",
                      "Precursor_MZ": "800"}
                 ]
          }
       }
    }

- `peaks_json` and `Precursor_MZ` are the only mandatory fields.
- `Precursor_MZ` can be a string of int or a string of float. i.e. "800" or "800.00"
- The optional `n_best_spectra` parameter controls the number of predicted spectra returned per set of peaks (10 by default).
- The optional `include_metadata` parameter controls the result spectra metadata returned to the user.

The available endpoints are:

- `spec2vec-positive`
- `spec2vec-negative`
- `ms2deepscore`

Black format your code
-------------------------------------

Please black format you code before checking in. This should be done using the black
version provided in the environment and the following command:
::

    black --target-version py37 omigami
