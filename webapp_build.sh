#!/bin/bash
set -e

IMAGE_NAME=${1:-riskoff-webapp}

if docker build -t ghcr.io/gdescamps/${IMAGE_NAME}:dev .; then
  docker push ghcr.io/gdescamps/${IMAGE_NAME}:dev
fi
