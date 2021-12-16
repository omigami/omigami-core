export OMIGAMI_ENV=dev
export STORAGE_ROOT=s3://omigami-dev

MODEL_RUN_ID="$1"
IMAGE="$2"
FLOW_NAME="$3"
ION_MODE="$4"


omigami spec2vec deploy-model \
    --model-run-id=$MODEL_RUN_ID \
    --image=$IMAGE \
    --project-name=spec2vec \
    --flow-name=$FLOW_NAME \
    --dataset-id=10k \
    --ion-mode=$ION_MODE