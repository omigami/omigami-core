snap ()
{
    echo "SNAPSHOT.$(git rev-parse --short HEAD)"
}

BASE_TAG="drtools/prefect:omigami-$(snap)"
S2V_TAG="drtools/omigami-spec2vec:$(snap)"
MS2DS_TAG="drtools/omigami-ms2deepscore:$(snap)"


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
  case $key in
      -p|--push)
      PUSH="true"
      shift # past argument
      ;;
      *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done
set -- "${POSITIONAL[@]}"


echo "$BASE_TAG"

docker build -t $BASE_TAG -f docker/spectra_matching.Dockerfile . --platform linux/amd64
docker build -t $S2V_TAG --build-arg BASE_IMAGE=$BASE_TAG -f docker/spec2vec.Dockerfile . --platform linux/amd64
docker build -t $MS2DS_TAG --build-arg BASE_IMAGE=$BASE_TAG -f docker/ms2deepscore.Dockerfile . --platform linux/amd64


if [[ -z "$PUSH" ]]; then
  echo "Will not push to the image registry."
else
  echo "Pushing images."
  docker push $BASE_TAG
  echo "Successfully pushed $BASE_TAG"
  docker push $MS2DS_TAG
  echo "Successfully pushed $MS2DS_TAG"
  docker push $S2V_TAG
  echo "Successfully pushed $S2V_TAG"
fi
