#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np
from scipy.signal import resample
from scipy.signal.windows import hann

from rknnlite.api import RKNNLite

from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.event import Event
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.info import Attribution, Describe, Info, AsrProgram, AsrModel

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("rknpu-whisper-wyoming")

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 20
MAX_LENGTH = CHUNK_LENGTH * 100
N_MELS = 80
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"
CHUNK_OVERLAP_SEC = 1.0
MIN_CHUNK_RMS = 0.003

END_TOKEN = 50257
SOT_TOKEN = 50258
TASK_FOR_EN = 50259
TASK_FOR_ZH = 50260
NO_TIMESTAMPS = 50363
TRANSCRIBE = 50359
TIMESTAMP_BEGIN = 50364


class WhisperRKNN:
    def __init__(self, model_dir: str, language: str = "en"):
        self.model_dir = Path(model_dir)
        self.language = language
        self.task_code = TASK_FOR_ZH if language == "zh" else TASK_FOR_EN

        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

        self.encoder_path, self.decoder_path = self._find_models()

        vocab_path = self.model_dir / ("vocab_zh.txt" if language == "zh" else "vocab_en.txt")
        mel_filters_path = self.model_dir / "mel_80_filters.txt"
        self.vocab = read_vocab(vocab_path)
        self.mel_filters = np.loadtxt(mel_filters_path, dtype=np.float32).reshape((N_MELS, N_FFT // 2 + 1))

        self.encoder = self._load_rknn(self.encoder_path)
        self.decoder = self._load_rknn(self.decoder_path)

        _LOGGER.info("Loaded encoder: %s", self.encoder_path)
        _LOGGER.info("Loaded decoder: %s", self.decoder_path)

    def _find_models(self):
        files = sorted(self.model_dir.glob("*.rknn"))
        encoder = next((p for p in files if "encoder" in p.name.lower()), None)
        decoder = next((p for p in files if "decoder" in p.name.lower()), None)
        if encoder is None or decoder is None:
            raise FileNotFoundError(f"Need encoder*.rknn and decoder*.rknn in model dir: {self.model_dir}")
        return encoder, decoder

    def _load_rknn(self, path: Path):
        rknn = RKNNLite()
        ret = rknn.load_rknn(str(path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {path}, ret={ret}")

        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {path}, ret={ret}")

        return rknn

    def transcribe_pcm(self, pcm_bytes: bytes, rate: int, width: int, channels: int) -> str:
        audio = pcm_to_float32_mono_16k(pcm_bytes, rate, width, channels)
        if audio.size == 0:
            return ""

        audio = condition_audio(audio)
        transcripts = []
        for chunk in split_audio(audio):
            text = self.transcribe_audio(chunk)
            if text:
                transcripts.append(text)

        return merge_transcripts(transcripts)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        mel = log_mel_spectrogram(audio, self.mel_filters)
        x_mel = pad_or_trim_mel(mel)[None, ...].astype(np.float32)

        out_encoder = self.encoder.inference(inputs=[x_mel], data_format="nchw")[0]
        text = run_decoder(self.decoder, out_encoder, self.vocab, self.task_code)

        return text.strip()

    def close(self):
        self.encoder.release()
        self.decoder.release()


class RknpuWhisperEventHandler(AsyncEventHandler):
    def __init__(self, reader, writer, model: WhisperRKNN):
        super().__init__(reader, writer)
        self.model = model
        self.audio = bytearray()
        self.rate = SAMPLE_RATE
        self.width = 2
        self.channels = 1

    async def handle_event(self, event: Event) -> bool:
        if event.type == Describe().event().type:
            await self.write_event(
                Info(
                    asr=[
                        AsrProgram(
                            name="rknpu-whisper",
                            description="Whisper RKNN/RKNPU Wyoming STT",
                            version=None,
                            attribution=Attribution(
                                name="Rockchip RKNN Whisper",
                                url="https://github.com/airockchip/rknn_model_zoo",
                            ),
                            installed=True,
                            models=[
                                AsrModel(
                                    name="whisper-rknn",
                                    description="Whisper RKNN encoder/decoder",
                                    version=None,
                                    attribution=Attribution(
                                        name="Rockchip RKNN Whisper",
                                        url="https://github.com/airockchip/rknn_model_zoo",
                                    ),
                                    installed=True,
                                    languages=["en", "zh"],
                                )
                            ],
                        )
                    ]
                ).event()
            )
            return True

        if Transcribe.is_type(event.type):
            self.audio.clear()
            return True

        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self.rate = audio_start.rate
            self.width = audio_start.width
            self.channels = audio_start.channels
            self.audio.clear()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.rate = chunk.rate
            self.width = chunk.width
            self.channels = chunk.channels
            self.audio.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            if not self.audio:
                await self.write_event(Transcript(text="").event())
                return True

            audio_seconds = len(self.audio) / max(1, self.rate * self.width * self.channels)
            _LOGGER.info(
                "Transcribing %.2f sec audio: rate=%s width=%s channels=%s",
                audio_seconds,
                self.rate,
                self.width,
                self.channels,
            )

            start_time = time.perf_counter()
            try:
                text = await asyncio.to_thread(
                    self.model.transcribe_pcm,
                    bytes(self.audio),
                    self.rate,
                    self.width,
                    self.channels,
                )
            except Exception:
                _LOGGER.exception("Transcription failed")
                text = ""

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            realtime_factor = elapsed_ms / max(1.0, audio_seconds * 1000)
            _LOGGER.info(
                "STT latency: %.0f ms for %.2f sec audio, realtime_factor=%.2fx",
                elapsed_ms,
                audio_seconds,
                realtime_factor,
            )
            _LOGGER.info("Transcript: %s", text)
            await self.write_event(Transcript(text=text).event())
            self.audio.clear()
            return True

        return True


def pcm_to_float32_mono_16k(pcm_bytes: bytes, rate: int, width: int, channels: int) -> np.ndarray:
    if rate <= 0:
        raise ValueError(f"Unsupported audio rate: {rate}")
    if channels <= 0:
        raise ValueError(f"Unsupported audio channels: {channels}")
    if not pcm_bytes:
        return np.empty(0, dtype=np.float32)

    usable_bytes = (len(pcm_bytes) // width) * width
    pcm_bytes = pcm_bytes[:usable_bytes]
    if not pcm_bytes:
        return np.empty(0, dtype=np.float32)

    if width == 2:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported audio width: {width}")

    if channels > 1:
        usable = (audio.size // channels) * channels
        audio = audio[:usable]
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != SAMPLE_RATE:
        desired_length = int(round(len(audio) / rate * SAMPLE_RATE))
        if desired_length <= 0:
            return np.empty(0, dtype=np.float32)
        audio = resample(audio, desired_length).astype(np.float32)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def condition_audio(audio: np.ndarray, target_rms: float = 0.08, max_gain: float = 6.0) -> np.ndarray:
    if audio.size == 0:
        return audio

    audio = audio - float(np.mean(audio))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))

    if rms > 1e-5 and peak < 0.98:
        gain = min(max_gain, max(1.0, target_rms / rms))
        audio = audio * gain

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def log_mel_spectrogram(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    pad_mode = "reflect" if audio.size > 1 else "constant"
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode=pad_mode)

    if len(padded) < N_FFT:
        padded = np.pad(padded, (0, N_FFT - len(padded)))

    frame_count = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    window = hann(N_FFT, sym=False).astype(np.float32)

    magnitudes = np.empty((N_FFT // 2 + 1, max(frame_count - 1, 1)), dtype=np.float32)

    for i in range(frame_count):
        if i >= magnitudes.shape[1]:
            break

        start = i * HOP_LENGTH
        frame = padded[start : start + N_FFT] * window
        spectrum = np.fft.rfft(frame, n=N_FFT)
        magnitudes[:, i] = np.abs(spectrum).astype(np.float32) ** 2

    mel_spec = mel_filters @ magnitudes
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)

    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def pad_or_trim_mel(audio_array: np.ndarray) -> np.ndarray:
    x_mel = np.zeros((N_MELS, MAX_LENGTH), dtype=np.float32)
    real_length = min(audio_array.shape[1], MAX_LENGTH)
    x_mel[:, :real_length] = audio_array[:, :real_length]
    return x_mel


def split_audio(audio: np.ndarray) -> list[np.ndarray]:
    max_samples = CHUNK_LENGTH * SAMPLE_RATE
    overlap = int(CHUNK_OVERLAP_SEC * SAMPLE_RATE)
    step = max_samples - overlap

    if audio.size <= max_samples:
        return [audio] if audio_rms(audio) >= MIN_CHUNK_RMS else []

    chunks = []
    start = 0
    while start < audio.size:
        end = min(start + max_samples, audio.size)
        chunk = audio[start:end]
        if audio_rms(chunk) >= MIN_CHUNK_RMS:
            chunks.append(chunk)

        if end >= audio.size:
            break

        start += step

    return chunks


def audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(audio))))


def read_vocab(vocab_path: Path) -> dict[str, str]:
    vocab = {}

    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) == 1:
                vocab[parts[0]] = ""
            else:
                vocab[parts[0]] = parts[1]

    return vocab


def run_decoder(
    decoder_model,
    out_encoder: np.ndarray,
    vocab: dict[str, str],
    task_code: int,
    max_decode_steps: int = 224,
) -> str:
    tokens = [SOT_TOKEN, task_code, TRANSCRIBE, NO_TIMESTAMPS]
    context_length = 12
    tokens_str = ""
    pop_id = context_length

    tokens = tokens * int(context_length / 4)
    next_token = SOT_TOKEN

    for _ in range(max_decode_steps):
        out_decoder = decoder_model.inference(
            inputs=[
                np.asarray([tokens], dtype=np.int64),
                out_encoder,
            ],
            data_format="nchw",
        )[0]

        next_token = select_next_token(out_decoder[0, -1])
        tokens.append(next_token)

        if next_token == END_TOKEN:
            tokens.pop(-1)
            break

        if next_token >= TIMESTAMP_BEGIN:
            tokens.pop(-1)
            continue

        if pop_id > 4:
            pop_id -= 1

        tokens.pop(pop_id)
        tokens_str += vocab.get(str(next_token), "")
    else:
        _LOGGER.warning("Decoder stopped after max_decode_steps=%s without end token", max_decode_steps)

    result = tokens_str.replace("\u0120", " ").replace("<|endoftext|>", "").replace("\n", "")

    if task_code == TASK_FOR_ZH:
        result = base64_decode(result)

    return result.strip()


def select_next_token(logits: np.ndarray) -> int:
    next_token = int(np.argmax(logits))
    if next_token < TIMESTAMP_BEGIN:
        return next_token

    masked_logits = np.array(logits, copy=True)
    if TIMESTAMP_BEGIN < masked_logits.size:
        masked_logits[TIMESTAMP_BEGIN:] = -np.inf

    for token in (SOT_TOKEN, TASK_FOR_EN, TASK_FOR_ZH, TRANSCRIBE, NO_TIMESTAMPS):
        if token < masked_logits.size:
            masked_logits[token] = -np.inf

    return int(np.argmax(masked_logits))


def merge_transcripts(transcripts: list[str]) -> str:
    merged = ""
    for transcript in transcripts:
        transcript = transcript.strip()
        if not transcript:
            continue

        if not merged:
            merged = transcript
            continue

        merged = merge_two_transcripts(merged, transcript)

    return merged.strip()


def merge_two_transcripts(left: str, right: str, max_overlap_chars: int = 80) -> str:
    normalized_left = left.lower()
    normalized_right = right.lower()
    max_overlap = min(len(left), len(right), max_overlap_chars)

    for overlap in range(max_overlap, 0, -1):
        if normalized_left[-overlap:] == normalized_right[:overlap]:
            return left + right[overlap:]

    separator = "" if left.endswith((" ", "\n")) or right.startswith((" ", "\n")) else " "
    return left + separator + right


def base64_decode(encoded_string: str) -> str:
    if not encoded_string:
        return ""

    padded = encoded_string + ("=" * (-len(encoded_string) % 4))
    try:
        return base64.b64decode(padded).decode("utf-8", errors="replace")
    except Exception:
        _LOGGER.warning("Failed to decode zh base64 transcript", exc_info=True)
        return encoded_string


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--language", default="en", choices=["en", "zh"])
    args = parser.parse_args()

    model = WhisperRKNN(args.model_dir, args.language)
    server = AsyncServer.from_uri(args.uri)

    _LOGGER.info("Starting Wyoming RKNN Whisper server on %s", args.uri)

    try:
        await server.run(partial(RknpuWhisperEventHandler, model=model))
    finally:
        model.close()


if __name__ == "__main__":
    asyncio.run(main())
