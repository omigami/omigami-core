# Deployment

## Choose the cluster

```shell
export OMIGAMI_ENV=dev  # Either 'dev' or 'prod'
```

## Where to find `S2V_IMAGE` `MS2DS_IMAGE` and `MODEL_RUN_ID`
* S2V_IMAGE: [omigami-spec2vec repository](https://hub.docker.com/repository/docker/drtools/omigami-spec2vec/tags?page=1&ordering=last_updated)
* MS2DS_IMAGE: [omigami-ms2deepscore repository](https://hub.docker.com/repository/docker/drtools/omigami-ms2deepscore/tags?page=1&ordering=last_updated)
* MODEL_RUN_ID (dev): https://dev.omigami.com/mlflow/#/
* MODEL_RUN_ID (prod): https://app.omigami.com/mlflow/#/

Then you can check flows here:
* (dev): https://prefect-dev.omigami.com/
* (prod): https://prefect-app.omigami.com/

### Spec2Vec

#### Training Flow

For information about all parameters and defaults:
```shell
omigami spec2vec train --help
```


```shell
omigami spec2vec train \ 
    --iterations=5 \
    # default (25) is suggested for prod deployment 
    --n-decimals=2 \
    --window=500 \
    --intensity-weighting-power=0.5 \
    --allowed-missing-percentage=5.0 \
    --image=<S2V_IMAGE> \
    --project-name=spec2vec \
    --flow-name=<FLOW_NAME> \
    --dataset-id=10k \
    # 'complete' is suggested for prod 
    --source-uri=https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json \
    # https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json is suggested for prod
    --ion-mode=positive \
    --deploy-model \
    --overwrite-model \
    --dataset-directory=s3://omigami-dev/datasets \
    --schedule=30

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
    # 'complete' is suggested for prod deployment
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
    --fingerprint-n-bits=2048 \
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
    # 'complete' is suggested for prod
    --source-uri=https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json \
    # https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json is suggested for prod
    --ion-mode=positive \
    --deploy-model \
    --overwrite-model \
    --dataset-directory=s3://omigami-dev/datasets \
    --schedule=30

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
    # 'complete' is suggested for prod deployment
    --ion-mode=positive
```

# How to access predictions
After the model has been deployed you can access the predictions endpoint in two ways:

By making a curl request
```shell
curl -v https://mlops.datarevenue.com/seldon/seldon/<endpoint-name>/api/v0.1/predictions -H "Content-Type: application/json" -d 'input_data'
```

```shell
curl -v https://mlops.datarevenue.com/seldon/seldon/<endpoint-name>/api/v0.1/predictions -H "Content-Type: application/json" -d @path_to/input.json
```

By accessing the external API with the user interface at:

```shell
https://mlops.datarevenue.com/seldon/seldon/<endpoint-name>/api/v0.1/doc/
```

Or by querying the prediction API via the python request library (see notebook in the client repository)


The input data should look like:

```shell
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
```

- `peaks_json` and `Precursor_MZ` are the only mandatory fields.
- `Precursor_MZ` can be a string of int or a string of float. i.e. "800" or "800.00"
- The optional `n_best` parameter controls the number of predicted spectra returned per set of peaks (10 by default).

The available endpoints are:

- `spec2vec-positive`
- `spec2vec-negative`
- `ms2deepscore`
