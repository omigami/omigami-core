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

```shell
omigami spec2vec deploy-model \
    --model-run-id=<MODEL_RUN_ID> \
    --n-decimals=2 \
    --intensity-weighting-power=0.5 \
    --allowed-missing-percentage=5.0 \
    --image=<S2V_IMAGE> \
    --project-name=spec2vec \
    --flow-name=<FLOW_NAME> \
    --dataset-id=10k \
    --ion-mode=positive
```



### MS2DeepScore

#### Training Flow

For information about all parameters and defaults:
```shell
omigami ms2deepscore train --help
```

```shell
omigami ms2deepscore train \ 
    ----fingerprint-n-bits=2048 \
    --scores-decimals=5 \
    --spectrum-binner-n-bins=10000 \
    --spectrum-ids-chunk-size=10000 \
    --train-ratio=0.9 \
    --validation-ratio=0.05 \
    --test-ratio=0.05 \
    --epochs=5.0 \
    --image=<MS2DS_IMAGE> \
    --project-name=ms2deepscore \
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
omigami ms2deepscore deploy-model --help
```

```shell
omigami spec2vec deploy-model \
    --model-run-id=<MODEL_RUN_ID> \
    --image=<MS2DS_IMAGE> \
    --project-name =ms2deepscore \
    --flow-name=<FLOW_NAME> \
    --dataset-id=10k \
    --ion-mode=positive
```