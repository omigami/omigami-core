Development Environment
=======================

How to setup
------------
::

    conda env create -f requirements/development/environment.frozen.yaml
    conda activate omigami
    pip install -e .

How to update all packages
--------------------------
To update all packages and create new frozen environments. Make sure you have
correct environment activated.  You'll need at least `conda>=4.9`::

    conda activate omigami
    python requirements/development/dress.py env freeze requirements/development/environment.yaml
    conda env update -f requirements/development/environment.frozen.yaml

How to add or update a single package
-------------------------------------

1. Install the package as usual with conda
2. Check which version was installed and add major and minor versions to
`requirements/development/environment.frozen.yaml`
3. Add the package with the most relaxed version restrictions possible to
`requirements/development/environment.yaml`
4. If the package is required to run any of the spectra_matching Prefect flows, add
the package to the correct text requirement file in the directory requirements/spectra_matching

E.g. if a package is required by ms2deepscore only and installed by conda, add it to
`requirements/spectra_matching/ms2deepscore/conda.txt`

Another example is, if a package is required by both ms2deepscore and spec2vec
and installed by pip, add it to
`requirements/spectra_matching/pip.txt`

Special steps for M1 Users
-------------------------------------

1. Activate the conda env and uninstall tensorflow through `pip uninstall tensorflow`
2. Go to https://drive.google.com/drive/folders/11cACNiynhi45br1aW3ub5oQ6SrDEQx4p (source: https://github.com/tensorflow/tensorflow/issues/46044#issuecomment-797151218) and download the `tensorflow-2.5.0-py3-none-any.whl` file.
3. Now do `pip install <path_to_downloaded_file>`
4. You might need to install manually one or two packages. Run the tests and if necessary (you are getting ModuleNotFound errors) install the missing packages.

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
    mlflow ui --backend-store-uri sqlite:///<PATH_TO_PROJECT_ROOT>/local-deployment/results/mlflow.sqlite


To access it: http://localhost:5000/


To run tests one by one via PyCharm, you can add this to your pytest Environment Variables (Run > Edit Configurations...)
::

    SKIP_REDIS_TEST=False;
    PREFECT__FLOWS__CHECKPOINTING=True;
    REDIS_DB=0;
    MLFLOW_SERVER=sqlite:///<absolute_path_to_project_root>/local-deployment/results/mlflow.sqlite;
    OMIGAMI_ENV=local

For the MLFLOW server path, you can get the correct value with sqlite:///$(pwd)/local-deployment/results/mlflow.sqlite
One example of MLFLOW_SERVER variable is (notice the 4 slashes):
::
    sqlite:////Users/czanella/dev/datarevenue/omigami-core/local-deployment/results/mlflow.sqlite

Running Prefect Tests using a built docker image
-----------------------------------------------------

Running flows in docker can be used to test images. To run in docker a few environment
variables must be changed, and a prefect docker agent must be used instead of a local one.

We first need to connect redis to prefect-server network and then spin up a docker agent.
Assuming prefect server is already up:
::
    docker network connect prefect-server redis
    prefect agent docker start -n local-docker-agent -l dev --show-flow-logs --log-level DEBUG --network prefect-server


Then a few environment variables must be updated on pytest settings:
::
    MLFLOW_SERVER=sqlite:///mlflow.sqlite;
    OMIGAMI_ENV=docker


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

PS: currently online endpoints are not available (server is down for indeterminate time)

Black format your code
-------------------------------------

Please black format you code before checking in. This should be done using the black
version provided in the environment and the following command:
::

    black --target-version py37 omigami


Please don't commit `*model.pkl` files to git. Every necessary model for the
test setup is going to be generated and saved to `test/assets/` folder and be
used from there on. You can also regenerate them at will if necessary (if you change some code that breaks the old pickled code).