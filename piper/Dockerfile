FROM --platform=$BUILDPLATFORM python:3.11-slim-bookworm AS models

ARG PIPER_MODEL_URL=https://huggingface.co/buckets/HanzoHuang/piper-voices-rknn-bucket/resolve/en/en_US/amy/medium
COPY piper/download_model.py /download_model.py
RUN python /download_model.py --url "$PIPER_MODEL_URL" --output /model

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        espeak-ng-data \
        libatomic1 \
        libdrm2 \
        libgomp1 \
        libsndfile1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY runtime/lib/librknnrt.so /usr/lib/librknnrt.so

COPY piper/requirements.txt /app/requirements.txt
COPY runtime/wheels/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /tmp/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
RUN python -m pip install --upgrade pip \
    && python -m pip install /tmp/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
    && python -m pip install -r /app/requirements.txt \
    && rm -f /tmp/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

COPY piper/wyoming_service.py /app/wyoming_service.py
COPY --from=models /model/encoder.onnx /app/model/encoder.onnx
COPY --from=models /model/decoder_rk3576.rknn /app/model/decoder_rk3576.rknn
COPY --from=models /model/config.json /app/model/config.json

ARG PIPER_VOICE=en_US-amy-medium
ENV PIPER_VOICE=$PIPER_VOICE

EXPOSE 10200

CMD ["python", "/app/wyoming_service.py", "--model-dir", "/app/model", "--uri", "tcp://0.0.0.0:10200"]
