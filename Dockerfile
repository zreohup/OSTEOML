FROM python:3.13-slim

ENV HOST=0.0.0.0 \
    PORT=7860 \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY web ./web

EXPOSE 7860

CMD ["python", "web/model_service.py"]
