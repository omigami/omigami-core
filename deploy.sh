snap ()
{
    echo "-SNAPSHOT.$(git rev-parse --short HEAD)"
}

#TODO: Make tag versionable
TAG="drtools/prefect:spec2vec_mlops$(snap)"
SPARK_TAG="drtools/prefect:spec2vec_mlops_spark$(snap)"

echo "$TAG"

docker build -t $TAG -f default.Dockerfile .
docker push $TAG

echo "Successfully pushed $TAG"

if [ -z "$1" = "spark"]; then
  echo "Building spark image"
  docker build -t$SPARK_TAG -f default.Dockerfile .
  docker push $SPARK_TAG

fi