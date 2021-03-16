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
    python dress.py env freeze requirements/environment.yaml
    conda env update -f requirements/environment_test.yaml

How to add or update a single package
-------------------------------------

1. Install the package as usual with conda
2. Check which version was installed and add major and minor version to environment.frozen.yaml
3. Add the package with the most relaxed version restrictions possible to environment.yaml

How to set up a local Feast
-------------------------------------
::

    git clone https://github.com/feast-dev/feast.git
    cd feast/infra/docker-compose
    cp .env.sample .env

And to run it:
::

    docker-compose pull && docker-compose up -d