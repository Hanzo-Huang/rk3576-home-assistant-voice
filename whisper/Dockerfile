FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libatomic1 \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY runtime/lib/librknnrt.so /usr/lib/librknnrt.so

COPY whisper/requirements.txt /app/requirements.txt
COPY runtime/wheels/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /tmp/rknn_toolkit_lite2.whl
RUN python -m pip install --upgrade pip \
    && python -m pip install /tmp/rknn_toolkit_lite2.whl \
    && python -m pip install -r /app/requirements.txt \
    && rm -f /tmp/rknn_toolkit_lite2.whl

COPY whisper/wyoming_service.py /app/wyoming_service.py
COPY whisper/model /app/model

EXPOSE 10300

CMD ["python", "/app/wyoming_service.py", "--model-dir", "/app/model", "--uri", "tcp://0.0.0.0:10300"]
