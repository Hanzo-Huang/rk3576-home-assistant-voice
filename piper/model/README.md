# Piper model files

Piper voices are hosted on Hugging Face and downloaded inside the Docker build,
not copied from this directory. No model binaries should be committed here.

The catalog in [`../models.json`](../models.json) selects the published voices:

- `en_US-amy-medium` (English, default)
- `zh_CN-huayan-medium` (Chinese Mandarin)

Each catalog URL is a direct download directory containing:

- `encoder.onnx`
- `decoder_rk3576.rknn`
- `config.json`

Only these three inference files are downloaded and copied into each image.
The decoder is exclusively for RK3576. Other decoders, training checkpoints,
datasets, and training scripts are not downloaded.

Add a catalog entry to build and publish another voice, or pass `PIPER_MODEL_URL`
and `PIPER_VOICE` as Docker build arguments. See the root README for examples.
