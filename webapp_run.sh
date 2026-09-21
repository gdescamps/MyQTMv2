#!/bin/bash
set -euo pipefail

IMAGE_NAME=${1:-riskoff-webapp}
CONTAINER_NAME=${CONTAINER_NAME:-riskoff_webapp}

# Mot de passe d'acces au dashboard : WEBAPP_PASSWORD (+ WEBAPP_SECRET optionnel)
# lus dans .env — on n'injecte que ces deux cles dans le conteneur, pas les
# identifiants Bourso / Gmail.
env_from_dotenv() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- | sed -E 's/^"(.*)"$/\1/'; }
WEBAPP_PASSWORD=${WEBAPP_PASSWORD:-$(env_from_dotenv WEBAPP_PASSWORD)}
WEBAPP_SECRET=${WEBAPP_SECRET:-$(env_from_dotenv WEBAPP_SECRET)}
if [[ -z "$WEBAPP_PASSWORD" ]]; then
  echo "WEBAPP_PASSWORD manquant : ajoute WEBAPP_PASSWORD=... dans .env (ou exporte-le)" >&2
  exit 1
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 8081:8081 \
  -e WEBAPP_PASSWORD="${WEBAPP_PASSWORD}" \
  -e WEBAPP_SECRET="${WEBAPP_SECRET}" \
  -v "$(pwd)/outputs:/app/outputs:ro" \
  -v "$(pwd)/logs:/app/logs:ro" \
  -v "$(pwd)/dashboard/assets:/app/dashboard/assets:ro" \
  ghcr.io/gdescamps/${IMAGE_NAME}:dev
