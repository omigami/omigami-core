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
    conda activate spec2vec-mlops

How to update all packages
--------------------------
To update all packages and create new frozen environments. Make sure you have correct
environment activated. You'll need at least `conda>=4.9`::

    conda activate spec2vec-mlops
    python dress.py env freeze environment.yaml
    conda env update -f environment_test.yaml

How to add or update a single package
-------------------------------------

1. Install the package as usual with conda
2. Check which version was installed and add major and minor version to environment.frozen.yaml
3. Add the package with the most relaxed version restrictions possible to environment.yaml
