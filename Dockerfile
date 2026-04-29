FROM nvidia/cuda:12.4.1-base-ubuntu22.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md /app/
COPY search /app/search
COPY src /app/src
COPY scripts /app/scripts
COPY benchmarks /app/benchmarks
COPY configs /app/configs
COPY docs /app/docs
COPY program /app/program

RUN pip install --no-cache-dir --break-system-packages torch torchvision faiss-cpu
RUN pip install --no-cache-dir --break-system-packages .

CMD ["python3", "scripts/run_experiment.py", "--help"]

