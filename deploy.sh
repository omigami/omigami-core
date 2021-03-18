#TODO: Make tag versionable
TAG="drtools/prefect:spec2vec_mlops_v4"

docker build -t $TAG -f default.Dockerfile .
docker push $TAG
