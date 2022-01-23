
Omigami on MLOps architecture
=============================

Introduction
------------

This repository contains the functionality necessary to train spec2vec and ms2deepscore machine learning models
using data from the online GNPS datasets. There are two available options:

a) Train the models using simple jupyter notebooks which outputs the trained model location (ideal for local machine training)

b) Use a prefect pipeline through a CLI interface to orchestrate each task in the training flow (suited for training with large dataset in production)

Setting up the environment
--------------------------

You should install the conda python package to create a conda environment using the env files supplied in this repository.
Once you have conda installed, you can execute:

    conda env create -f requirements/development/environment.frozen.yaml 
    conda activate 
    omigami pip install -e .

to create a new conda environment called `omigami` and install the relevant packages for development.


If you want to build your own predicting scripts, there are some additional packages that may be useful to you and that can be installed 
by issuing the following commands with the new conda env activated:

    conda env update -f requirements/spectra_matching/spec2vec/conda.txt
    pip install -r requirements/spectra_matching/spec2vec/pip.txt

In this example, we are installing spec2vec spectra_matching dependencies. These packages are not necessary for training.

**Considering the tensorflow extra package from spectra_matching/ms2deepscore, for Apple Silicon (M1) users, please do the 
extra following steps to make sure tensorflow is installed correctly (only necessary for ms2deepscore users):**

1. Activate the conda env and uninstall tensorflow through `pip uninstall tensorflow`
2. Go to https://drive.google.com/drive/folders/11cACNiynhi45br1aW3ub5oQ6SrDEQx4p (source: https://github.com/tensorflow/tensorflow/issues/46044#issuecomment-797151218) and download the `tensorflow-2.5.0-py3-none-any.whl` file.
3. Now do `pip install <path_to_downloaded_file>`
4. You might need to install manually one or two packages. Run the tests and if necessary (you are getting ModuleNotFound errors) install the missing packages.


Using the Notebooks
-------------------

Go to `notebooks/training` to find the training notebook. Inside each notebook you can find instructions on how to set
the parameters to each training function.


Using the Prefect training flows (for power users)
--------------------------------------------------
For running the prefect training flows we need to setup local prefect and mlflow instances. Docker is also necessary
to achieve this local environment setup. You can install `docker desktop` for your machine following the guides at
www.docker.com. You must also have a dockerhub account, so you can build and push images to a repository owned by this
account. Look at next section to see how to build and push images (you will need to do it once, if you don't change any
code inside the `omigami` folder). Alternatively, you can use one of DataRevenue's public images, but having your own
image guarantees you are self suficient to run your training flows in the future.

Prefect is a task orchestrator, each training flow is consisted of multiple tasks that depend on each other in the form
of a DAG (directed acyclic graph). One task may, for example, download the GNPS dataset, while another, more complex task,
may run multiple cleaning task instances in parallel to prepare the data for training.

If you whish to experiment with the algorithms in smaller scale we recommend to use the jupyter notebooks instead.

You must initialize the prefect docker setup by doing `docker-compose -f local-deployment/docker-compose.yml up -d`

If you have any trouble, check first if you need to increase your docker memory limits. Do `docker ps` to see the containers that are
currently running.

Now, we will create a prefect tenant and start a prefect agent.

```
    prefect backend server
    prefect server create-tenant -n default    
    prefect agent start -l "dev" --show-flow-logs
```

The prefect agent is responsible to run the flows, and the argument `-l` in the last command specifies which flow tags 
the agent should catch for running. The `dev` tag is hardcoded in the training flows, so you should leave that argument
there for things to work.

Now you can checkout the prefect dashboard by going to `http://localhost:8080`.

If you want, you can familiarize yourself with the Prefect dashboards by taking a look at the prefect documentation at
https://docs.prefect.io/orchestration/ui/dashboard.html

To shut prefect down, use 
```
docker-compose -f local-deployment/docker-compose.yml down
```

Now, to run mlflow locally run the following command:
```
    mlflow ui --backend-store-uri sqlite:///<PATH_TO_PROJECT_ROOT>/local-deployment/results/mlflow.sqlite
```


To access it: http://localhost:5000/

Again, you can look at mlflow documentation to familirize yourself with its UI: https://www.mlflow.org/docs/latest/tracking.html#tracking-ui

Now you are ready to run some training flows!

To start a training flow, issue for instance:

```
export OMIGAMI_ENV=local
export MLFLOW_SERVER="sqlite:///$(pwd)/local-deployment/results/mlflow.sqlite"

omigami ms2deepscore train \
    --image "<image>" \
    --project-name ms2deepscore \ 
    --flow-name ms2deepscore-first-flow-test \ 
    --dataset-id small_500 \
    --ion-mode positive \ 
```

or

```
omigami spec2vec train \
    --flow-name spec2vec-training \
    --dataset-id small_500 \
    --ion-mode positive \
    --local
```

In the notebooks you can check all available options for each training method (you can alternatively check the `cli.py`
code inside each algorithm folder, nested inside the `omigami` source code folder)

Also, there is an environment variable called `STORAGE_ROOT` that can be used if you want to change where results and 
files from intermediate steps are stored.

Building and pushing images (for power users)
--------------------------------------------------
To run a training flow you will need to use a docker image containing the flow code and its dependencies. You can
build an image by issuing the following commands:
```
    docker build . -t <image>
```
Where `<image>` is a name that should follow the convention `yourrepository/yourimagename:yourtag`

After the image is built successfully, you can push the image to the remote repository 
(must be a public repository to be downloaded by the training flow):
```
    docker push <image>
```


If you wish, available public images from DataRevenue (most up to date) are:

`drtools/omigami-ms2deepscore:0.2.0-SNAPSHOT.6eaec921`
For ms2deepscore and

`drtools/omigami-spec2vec:0.2.0-SNAPSHOT.d7b9d604`
for spec2vec


Additional docs
-------------------
Additional docs for developers can be found in the readme inside the `omigami` folder.