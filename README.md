# Home Assistant Voice on RK3576

A local Home Assistant voice stack for Rockchip RK3576:

- Whisper speech-to-text through Wyoming
- Piper text-to-speech through Wyoming
- openWakeWord wake-word detection
- Qwen 2.5 1.5B through an OpenAI-compatible RKLLM API

Whisper and Piper use the RK3576 NPU. All images are prebuilt for Linux ARM64, so users do not need to build anything locally.

## Wyoming

[Wyoming](https://github.com/rhasspy/wyoming) is the protocol Home Assistant uses to communicate with local voice services. This stack exposes:

| Service | Port |
| --- | ---: |
| Piper TTS | `10200` |
| Whisper STT | `10300` |
| openWakeWord | `10400` |
| RKLLM API | `8001` |

## Start

Install Docker with the Compose plugin on the RK3576 board, clone this repository, and run:

```bash
sudo docker compose up -d --pull always
```

Check the services:

```bash
sudo docker compose ps
sudo docker compose logs -f
```

Stop the services:

```bash
sudo docker compose down
```

The default stack assumes Home Assistant is already running on another machine.

To also run Home Assistant on the RK3576 board:

```bash
sudo docker compose --profile homeassistant up -d --pull always
```

Open `http://RK3576_BOARD_IP:8123` to finish its setup.

## Configure Home Assistant

In Home Assistant, go to **Settings -> Devices & services -> Add integration** and add the **Wyoming Protocol** integration for each service:

| Service | Host | Port |
| --- | --- | ---: |
| Whisper STT | RK3576 board IP | `10300` |
| Piper TTS | RK3576 board IP | `10200` |
| openWakeWord | RK3576 board IP | `10400` |

Next, go to **Settings -> Voice assistants**, create or edit an Assist pipeline, and select those services.

### Local LLM

Install the [Local LLM integration](https://github.com/acon96/home-llm) through HACS and configure it with:

```text
Backend: OpenAI Compatible Conversations API
API hostname: RK3576_BOARD_IP
API port: 8001
API path: /v1
API key: sk-local
Model name: rkllm-model
```

Select the new conversation agent in the Assist pipeline. The API key is only a placeholder for the local server.

The default LLM is Qwen 2.5 1.5B. To choose another RK3576 model, find an image in the [`rkllm-docker` repository](https://github.com/hanzo-huang/rkllm-docker) and replace the `llm.image` value in `docker-compose.yml`.

## Requirements

- Rockchip RK3576 board running Linux ARM64
- Docker Engine with Docker Compose
- Access to `/dev/rknpu` and `/dev/dma_heap`
