#/bin/bash

export $(grep -v '^#' .env | xargs)
docker build --build-arg username=$USER_NAME --build-arg password=$PASSWORD --build-arg repo_url=$REPO_URL -t full_tracker_image .