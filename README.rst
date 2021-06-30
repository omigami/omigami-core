##############################
Spec2Vec on MLOps architecture
##############################

Development Environment
=======================

How to setup
------------
::

    conda env create -f requirements/environment.frozen.yaml
    conda env update -f requirements/environment_test.yaml
    conda activate omigami
    pip install -e .

How to update all packages
--------------------------
To update all packages and create new frozen environments. Make sure you have correct
environment activated. You'll need at least `conda>=4.9`::

    conda activate omigami
    python requirements/dress.py env freeze requirements/environment.yaml
    conda env update -f requirements/environment.frozen.yaml
    conda env update -f requirements/environment_test.yaml

How to add or update a single package
-------------------------------------

1. Install the package as usual with conda
2. Check which version was installed and add major and minor version to environment.frozen.yaml
3. Add the package with the most relaxed version restrictions possible to environment.yaml

How to build a docker image
-------------------------------------
Whenever a Flow is registered in Prefect Server using Kubernetes, it needs to use a
Docker image that has Prefect installed and all the packages needed for running the code.

Usually, updating the requirements file should cover all the needed packages. Otherwise,
you should update the Dockerfile.

In order to publish this Docker Image, there's an auxiliary script to do this.
To run it, execute::

    bash deploy.sh

How to set up a local Feast
-------------------------------------
::

    git clone https://github.com/feast-dev/feast.git
    cd feast/infra/docker-compose
    cp .env.sample .env

And to run it:
::

    docker-compose pull && docker-compose up -d

How to run tests that require Redis locally
-------------------------------------------

Some tests that use feature store requires Redis to run.
Start a Redis container and set these environment variables before running the test suite:
::

    docker run -d --rm --name redis -p 6379:6379 redis:5-alpine
    export SKIP_REDIS_TEST=False

To run tests one by one via PyCharm, you can add this to your pytest Environment Variables (Run > Edit Configurations...)
::

    SKIP_REDIS_TEST=False;PREFECT__FLOWS__CHECKPOINTING=True;REDIS_HOST=localhost;REDIS_DB=0

How to register the training flow manually
------------------------------------------

To register a flow manually to Prefect you need to follow these steps:
::

    conda activate omigami
    export AWS_PROFILE=<your data revenue profile>
    export PYTHONPATH=$(pwd)
    prefect backend server
    python omigami/<tool_name>/cli.py register-training-flow -i [image] [options]

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