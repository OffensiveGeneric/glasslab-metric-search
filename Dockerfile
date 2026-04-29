FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y curl g++ && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

RUN pip install --no-cache-dir --break-system-packages . \
    && pip install --no-cache-dir --break-system-packages syne-tune 'ray[tune]' transformers

COPY search /app/search
COPY src /app/src
COPY scripts /app/scripts
COPY benchmarks /app/benchmarks
COPY configs /app/configs
COPY docs /app/docs
COPY program /app/program

ENV PYTHONPATH=/app

CMD ["python3", "scripts/run_experiment.py", "--help"]

