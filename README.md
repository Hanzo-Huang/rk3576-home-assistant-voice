# Home Assistant Voice on RK3576

A local voice stack for running Home Assistant Assist on a Rockchip RK3576 board.

This project packages the pieces Home Assistant needs for a private, local voice assistant:

- Speech-to-text with Whisper through the Wyoming protocol
- Text-to-speech with Piper through the Wyoming protocol
- Wake-word detection with openWakeWord
- Local conversation handling with Qwen 2.5 1.5B through an OpenAI-compatible RKLLM API

Whisper, Piper, and the LLM run on the RK3576 NPU. The Docker images are published for Linux ARM64, so normal users can pull and run them without building the models locally.

## What You Need

- A Rockchip RK3576 board running Linux ARM64
- Docker Engine with the Docker Compose plugin
- Access to the RK3576 device nodes, especially `/dev/rknpu` and `/dev/dma_heap`
- A Home Assistant instance on the same network, or the optional Home Assistant container in this compose file

## Quick Start

On the RK3576 board:

```bash
git clone https://github.com/Hanzo-Huang/rk3576-home-assistant-voice.git
cd rk3576-home-assistant-voice
```

Choose one start command:

```bash
# Voice stack only. Use this if Home Assistant runs elsewhere.
sudo docker compose up -d --pull always

# Voice stack plus Home Assistant on this RK3576 board.
sudo docker compose --profile homeassistant up -d --pull always
```

Check status:

```bash
sudo docker compose ps
sudo docker compose logs -f
```

If you started Home Assistant here, open:

```text
http://RK3576_BOARD_IP:8123
```

Use the RK3576 board IP address when adding services in Home Assistant.

## Services

The compose stack exposes these local services:

| Service | Purpose | Port |
| --- | --- | ---: |
| Piper | Text-to-speech | `10200` |
| Whisper | Speech-to-text | `10300` |
| openWakeWord | Wake-word detection | `10400` |
| RKLLM API | Local LLM, OpenAI-compatible API | `8001` |

## Configure Home Assistant

### 1. Add the Wyoming services

In Home Assistant:

1. Open **Settings -> Devices & services**.
2. Select **Add integration**.
3. Search for **Wyoming Protocol**.
4. Add each service below, using the RK3576 board IP address as the host.

| Service | Host | Port |
| --- | --- | ---: |
| Whisper STT | RK3576 board IP | `10300` |
| Piper TTS | RK3576 board IP | `10200` |
| openWakeWord | RK3576 board IP | `10400` |

### 2. Create an Assist pipeline

In Home Assistant:

1. Open **Settings -> Voice assistants**.
2. Create a new Assist pipeline, or edit an existing one.
3. Select the Wyoming Whisper service for speech-to-text.
4. Select the Wyoming Piper service for text-to-speech.
5. Select the Wyoming openWakeWord service for wake-word detection.

At this point, Home Assistant can use the local speech services.

### 3. Install HACS if needed

If HACS is already installed in Home Assistant, skip this step.

If you are running Home Assistant from this compose stack, install HACS inside the Home Assistant container:

```bash
sudo docker compose exec homeassistant bash -c "wget -O - https://get.hacs.xyz | bash -"
sudo docker compose restart homeassistant
```

For other Home Assistant installation types, follow the [HACS download guide](https://www.hacs.xyz/docs/use/download/download/).

After Home Assistant restarts, open **Settings -> Devices & services -> Add integration**, search for **HACS**, and complete its setup.

### 4. Add the local LLM

To use the RK3576 LLM as the conversation agent, add the Local LLM integration through HACS:

[![Open Local LLM in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=home-llm&owner=acon96)

Source repository: [acon96/home-llm](https://github.com/acon96/home-llm)

Configure it with:

```text
Backend: OpenAI Compatible Conversations API
API hostname: RK3576_BOARD_IP
API port: 8001
API path: /v1
API key: sk-local
Model name: rkllm-model
```

The API key is only a placeholder for the local server.

Then return to **Settings -> Voice assistants**, edit the Assist pipeline, and select the new local conversation agent.

## Stop or Update

Stop the stack:

```bash
sudo docker compose down
```

Pull newer images and restart:

```bash
sudo docker compose pull
sudo docker compose up -d
```

View logs for one service:

```bash
sudo docker compose logs -f whisper
sudo docker compose logs -f piper
sudo docker compose logs -f openwakeword
sudo docker compose logs -f llm
```

## Customize

### Change the wake word

The default openWakeWord model is `ok_nabu`.

To change it, edit the `openwakeword` command in [`docker-compose.yml`](docker-compose.yml):

```yaml
command:
  - --uri
  - tcp://0.0.0.0:10400
  - --preload-model
  - ok_nabu
```

Replace `ok_nabu` with the openWakeWord model you want to preload.

### Change the LLM

The default LLM image is:

```text
ghcr.io/hanzo-huang/rkllm-docker/qwen2.5-1.5b-instruct:w4a16-rk3576
```

To use another RK3576-compatible model, choose an image from the [`rkllm-docker` repository](https://github.com/hanzo-huang/rkllm-docker) and replace the `llm.image` value in [`docker-compose.yml`](docker-compose.yml).

Restart after editing:

```bash
sudo docker compose up -d
```

### Change Whisper language

Whisper supports English and Chinese model vocabularies in this image. The default service starts with English.

To use Chinese, override the Whisper command in [`docker-compose.yml`](docker-compose.yml):

```yaml
whisper:
  image: ghcr.io/hanzo-huang/wyoming-whisper-rk3576:latest
  restart: unless-stopped
  privileged: true
  ports:
    - "10300:10300"
  command:
    - python
    - /app/wyoming_service.py
    - --model-dir
    - /app/model
    - --uri
    - tcp://0.0.0.0:10300
    - --language
    - zh
```

Restart after editing:

```bash
sudo docker compose up -d
```

## Troubleshooting

### A service keeps restarting

Check its logs first:

```bash
sudo docker compose logs -f whisper
```

If Whisper, Piper, or the LLM cannot access the NPU, confirm the board exposes the expected devices:

```bash
ls -l /dev/rknpu /dev/dma_heap
```

### Home Assistant cannot connect

Confirm the containers are running:

```bash
sudo docker compose ps
```

Make sure Home Assistant uses the RK3576 board IP address, not `localhost`, unless Home Assistant is running on the same board with host networking.

### The LLM integration does not respond

Check the LLM container logs:

```bash
sudo docker compose logs -f llm
```

Then confirm the Local LLM integration uses:

```text
API path: /v1
API port: 8001
Model name: rkllm-model
```

## Development Notes

The Whisper and Piper images are built by GitHub Actions. During image builds, the workflow downloads the RK3576 model archives from a release bundle and places them in:

- [`whisper/model`](whisper/model)
- [`piper/model`](piper/model)

The model assets are intentionally not committed to the repository. See the model directory READMEs for the expected files.

## Related Projects

- [Wyoming protocol](https://github.com/rhasspy/wyoming)
- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [Local LLM integration for Home Assistant](https://github.com/acon96/home-llm)
- [RKLLM Docker images](https://github.com/hanzo-huang/rkllm-docker)
