#!/bin/bash

CONTAINER_NAME=${CONTAINER_NAME:-riskoff_webapp}

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}
