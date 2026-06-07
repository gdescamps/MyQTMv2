#!/bin/bash
set -euo pipefail

IMAGE_NAME=${1:-riskoff-webapp}
CONTAINER_NAME=${CONTAINER_NAME:-riskoff_webapp}

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 8081:8081 \
  -v "$(pwd)/outputs:/app/outputs:ro" \
  -v "$(pwd)/logs:/app/logs:ro" \
  -v "$(pwd)/dashboard/assets:/app/dashboard/assets:ro" \
  ghcr.io/gdescamps/${IMAGE_NAME}:dev
