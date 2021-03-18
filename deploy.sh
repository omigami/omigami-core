#TODO: Make tag versionable
TAG="drtools/prefect:spec2vec_mlops_v4"
SPARK_TAG="drtools/prefect:spec2vec_mlops_spark_v1"


docker build -t $TAG -f default.Dockerfile .
docker push $TAG

docker build -t$SPARK_TAG -f default.Dockerfile .
docker push $SPARK_TAG
