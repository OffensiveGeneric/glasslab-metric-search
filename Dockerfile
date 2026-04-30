FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ARG GLASSLAB_IMAGE_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    GLASSLAB_IMAGE_COMMIT=${GLASSLAB_IMAGE_COMMIT}

COPY pyproject.toml README.md requirements-runtime.txt /app/
COPY search /app/search
COPY src /app/src
COPY scripts /app/scripts
COPY benchmarks /app/benchmarks
COPY configs /app/configs
COPY docs /app/docs
COPY program /app/program

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.txt \
    && python -m pip uninstall -y ninja \
    && python -m pip install --no-deps .

CMD ["python3", "scripts/run_experiment.py", "--help"]
