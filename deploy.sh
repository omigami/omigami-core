set -e
function usage()
{
   cat << HEREDOC

   Usage: $(basename $0) TAG [-h] [-s] [-r REPO] [-p] [-j] [--rebuild-base]

   Tags the current HEAD as a release and pushes the tag. Next a docker image is built
   and optionally pushed. Snapshot releases can be built for testing with the -s option.

   optional arguments:
     -h, --help                show this help message and exit
     -s, --snapshot            marks this release as a snapshot release, will append
                              '-SNAPSHOT.[GIT-HASH]' to the TAG and not create a git tag.
     -p, --push                if set the image will also be pushed

HEREDOC
}



snap ()
{
    echo "-SNAPSHOT.$(git rev-parse --short HEAD)"
}


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
  case $key in
      -p|--push)
      PUSH="true"
      shift # past argument
      ;;
      -s|--snapshot)
      SNAPSHOT="true"
      shift # past argument
      ;;
      *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

TAG="$1"

if [[ -z "$SNAPSHOT" ]]; then
  echo ""
  echo "======================================="
  echo "Creating full release for v$TAG."
  echo "======================================="
  if [ "$(git describe --tags --abbrev=0)" != "$TAG" ]; then
    git tag "$1"
  fi
  git push --tags
else
  TAG="$TAG$(snap)"
  echo "Creating snapshot release $TAG"
fi


BASE_TAG="drtools/prefect:omigami-$TAG"
S2V_TAG="drtools/omigami-spec2vec:$TAG"
MS2DS_TAG="drtools/omigami-ms2deepscore:$TAG"


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
