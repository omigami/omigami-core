snap ()
{
    echo "-SNAPSHOT.$(git rev-parse --short HEAD)"
}

TAG="drtools/prefect:spec2vec_mlops$(snap)"
SPARK_TAG="drtools/prefect:spec2vec_mlops_spark$(snap)"
CREATE_SPARK = 0

echo "$TAG"

docker build -t $TAG -f default.Dockerfile .
docker push $TAG

echo "Successfully pushed $TAG"

while [ "$1" != "" ]; do
    case $1 in
        -s | --spark )          shift
                                CREATE_SPARK=1
                                ;;
    esac
    shift
done

if [[ CREATE_SPARK == 1 ]]; then
  echo "Building spark image"
  docker build -t$SPARK_TAG -f spark.Dockerfile .
  docker push $SPARK_TAG
  echo "Successfully pushed $SPARK_TAG"
fi