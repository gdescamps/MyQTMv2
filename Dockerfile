FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8081

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements_webapp.txt ./
RUN pip install --upgrade pip && pip install -r requirements_webapp.txt

COPY src/webapp.py ./src/webapp.py
COPY dashboard/assets ./dashboard/assets
COPY logs ./logs

RUN chown -R app:app /app

USER app

EXPOSE 8081
CMD ["python", "src/webapp.py"]
