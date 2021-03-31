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
    conda activate spec2vec_mlops

How to update all packages
--------------------------
To update all packages and create new frozen environments. Make sure you have correct
environment activated. You'll need at least `conda>=4.9`::

    conda activate spec2vec_mlops
    python requirements/dress.py env freeze requirements/environment.yaml
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

How to register the training flow manually
------------------------------------------

To register the flow manually to Prefect you need to follow these steps:
::

    conda activate spec2vec_mlops
    export AWS_PROFILE=<your data revenue profile>
    export PYTHONPATH=$(pwd)
    prefect backend server
    python spec2vec_mlops/flows/training_flow.py register-train-pipeline

Then you can check the flow here: https://prefect.mlops.datarevenue.com/default

Black format your code
-------------------------------------

Please black format you code before checking in. This should be done using the black
version provided in the environment and the following command:
::

    black --target-version py37 spec2vec_mlops