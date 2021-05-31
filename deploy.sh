snap ()
{
    echo "-SNAPSHOT.$(git rev-parse --short HEAD)"
}

TAG="drtools/prefect:omigami$(snap)"

echo "$TAG"

docker build -t $TAG -f default.Dockerfile .
docker push $TAG

echo "Successfully pushed $TAG"
