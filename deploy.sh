snap ()
{
    echo "-SNAPSHOT.$(git rev-parse --short HEAD)"
}

TAG="drtools/prefect:spec2vec_mlops$(snap)"

echo "$TAG"

docker build -t $TAG -f default.Dockerfile .
docker push $TAG

echo "Successfully pushed $TAG"
