# Deployment

## Choose the cluster

```shell
export OMIGAMI_ENV=dev  # Either 'dev' or 'prod'
```

## Where to find `S2V_IMAGE` `MS2DS_IMAGE` and `MODEL_RUN_ID`
* S2V_IMAGE: [omigami-spec2vec repository](https://hub.docker.com/repository/docker/drtools/omigami-spec2vec/tags?page=1&ordering=last_updated)
* MS2DS_IMAGE: [omigami-ms2deepscore repository](https://hub.docker.com/repository/docker/drtools/omigami-ms2deepscore/tags?page=1&ordering=last_updated)
* MODEL_RUN_ID (dev): https://dev.omigami.com/mlflow/#/
* MODEL_RUN_ID (prod): TODO

### Spec2Vec

#### Training Flow

For information about all parameters and defaults:
```shell
omigami spec2vec train --help
```

Add example command here using all parameters

```shell
omigami spec2vec train \ 
    --iterations=5 \
    --n-decimals=2 \
    --window=500 \
    --intensity-weighting-power=0.5 \
    --allowed-missing-percentage=5.0 \
    --image=<S2V_IMAGE> \
    --project-name=spec2vec \
    --flow-name=<FLOW_NAME> \
    --dataset-id=10k \
    --source-uri=https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json \
    --ion-mode=positive \
    --deploy-model \
    --overwrite-model \
    --dataset-directory=s3://omigami-dev/datasets

```

#### Model Deployment flow

For information about all parameters and defaults:
```shell
omigami spec2vec deploy-model --help
```

Add example command here using all parameters
```shell
omigami spec2vec deploy-model \
    --model-run-id=<MODEL_RUN_ID> \
    --n-decimals=2 \
    --intensity-weighting-power=0.5 \
    --allowed-missing-percentage=5.0 \
    --image <S2V_IMAGE> \
    --project-name spec2vec \
    --flow-name=<FLOW_NAME> \
    --dataset-id=10k \
    --ion-mode positive
```



### MS2DeepScore

repeat