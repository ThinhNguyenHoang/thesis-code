export PROJECT_ID=thesis-372616
export REPO_NAME=thesis-registry
export IMAGE_NAME=base-cuda
export IMAGE_TAG=tensorflow-opencv
export IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

docker build -f Dockerfile -t ${IMAGE_URI} ./