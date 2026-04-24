FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY search /app/search
COPY src /app/src
COPY scripts /app/scripts
COPY benchmarks /app/benchmarks
COPY configs /app/configs
COPY docs /app/docs
COPY program /app/program

RUN pip install --no-cache-dir .

CMD ["python3", "scripts/run_experiment.py", "--help"]

