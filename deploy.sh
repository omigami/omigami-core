snap ()
{
    echo "-SNAPSHOT.$(git rev-parse --short HEAD)"
}

TAG="drtools/prefect:omigami$(snap)"

echo "$TAG"

docker build -t $TAG -f default.Dockerfile . --platform linux/amd64
docker push $TAG

echo "Successfully pushed $TAG"
