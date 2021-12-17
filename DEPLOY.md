# Deployment

## Choose the cluster

```shell
export OMIGAMI_ENV=dev  # Either 'dev' or 'prod'
```

## Spec2Vec

### Training Flow

For information about all parameters and defaults:
```shell
omigami spec2vec train --help
```

Add example command here using all parameters

```shell
omigami spec2vec train \ 
    --iterations 5 \
    --window 500 \
    --image <IMAGE> \
    --project-name spec2vec \
    --flow-name seldon-test-fix-small-ds \
    --dataset-id small \
    --deploy-model \
    --overwrite-model \
    --ion-mode positive
```

### Model Deployment flow

For information about all parameters and defaults:
```shell
omigami spec2vec deploy-model --help
```

Add example command here using all parameters
```shell
omigami spec2vec deploy-model \
    --model-run-id b2698d2a977344b3abe1d65079406a3e \
    --image <IMAGE> \
    --project-name spec2vec \
    --flow-name seldon-test-fix-small-ds-deploy \
    --dataset-id small \
    --ion-mode positive
```



## MS2DeepScore

repeat