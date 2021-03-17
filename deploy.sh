#TODO: Make tag versionable
TAG="drtools/prefect:spec2vec_mlops"

docker build -t $TAG -f default.Dockerfile .
docker push $TAG