#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
import wave
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime
from piper import PiperConfig, PiperVoice, SynthesisConfig
from piper.phonemize_espeak import ESPEAK_DATA_DIR
from rknnlite.api import RKNNLite
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("rknpu-piper-wyoming")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_URI = "tcp://0.0.0.0:10200"
DEFAULT_SAMPLES_PER_CHUNK = 1024


class PiperVoiceRKNN(PiperVoice):
    def __init__(
        self,
        *args,
        decoder_rknn: RKNNLite,
        decoder_chunk_size: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_rknn = decoder_rknn
        self.decoder_chunk_size = decoder_chunk_size

    @classmethod
    def load_from_dir(
        cls,
        model_dir: str | Path,
        use_cuda: bool = False,
        espeak_data_dir: str | Path = ESPEAK_DATA_DIR,
    ) -> "PiperVoiceRKNN":
        model_dir = Path(model_dir)
        encoder_path, decoder_path, config_path = find_model_files(model_dir)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        encoder_session = onnxruntime.InferenceSession(str(encoder_path), providers=providers)

        decoder_rknn = RKNNLite()
        ret = decoder_rknn.load_rknn(str(decoder_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {decoder_path}, ret={ret}")

        ret = decoder_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {decoder_path}, ret={ret}")

        with config_path.open("r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        decoder_chunk_size = get_decoder_chunk_size(decoder_rknn)

        _LOGGER.info("Loaded Piper encoder: %s", encoder_path)
        _LOGGER.info("Loaded Piper decoder: %s", decoder_path)
        _LOGGER.info("Loaded Piper config: %s", config_path)
        _LOGGER.info("RKNN decoder chunk size: %s", decoder_chunk_size)

        return cls(
            config=PiperConfig.from_dict(config_dict),
            session=encoder_session,
            decoder_rknn=decoder_rknn,
            decoder_chunk_size=decoder_chunk_size,
            espeak_data_dir=Path(espeak_data_dir),
        )

    def phoneme_ids_to_audio(
        self,
        phoneme_ids: list[int],
        syn_config: Optional[SynthesisConfig] = None,
        include_alignments: bool = False,
    ):
        if syn_config is None:
            syn_config = SynthesisConfig()

        speaker_id = syn_config.speaker_id
        length_scale = self.config.length_scale if syn_config.length_scale is None else syn_config.length_scale
        noise_scale = self.config.noise_scale if syn_config.noise_scale is None else syn_config.noise_scale
        noise_w_scale = self.config.noise_w_scale if syn_config.noise_w_scale is None else syn_config.noise_w_scale

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w_scale], dtype=np.float32)

        encoder_inputs = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
        }

        if self.config.num_speakers <= 1:
            speaker_id = None
        elif speaker_id is None:
            speaker_id = 0

        if speaker_id is not None:
            encoder_inputs["sid"] = np.array([speaker_id], dtype=np.int64)

        encoder_output = self.session.run(None, encoder_inputs)
        if speaker_id is not None:
            z, y_mask, g = encoder_output[:3]
        else:
            z, y_mask = encoder_output[:2]
            g = None

        audio_chunks = []
        for z_chunk, y_chunk, real_size in chunk_decoder_inputs(z, y_mask, self.decoder_chunk_size):
            decoder_inputs = [z_chunk.astype(np.float32), y_chunk.astype(np.float32)]
            if g is not None:
                decoder_inputs.append(g.astype(np.float32))

            decoder_output = self.decoder_rknn.inference(inputs=decoder_inputs, data_format="nchw")
            audio_chunk = trim_decoder_output(decoder_output[0], real_size, self.decoder_chunk_size)
            audio_chunks.append(audio_chunk)

        if not audio_chunks:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.concatenate(audio_chunks, axis=-1).squeeze().astype(np.float32)

        if include_alignments:
            return audio, None

        return audio

    def close(self) -> None:
        self.decoder_rknn.release()


class RknpuPiperEventHandler(AsyncEventHandler):
    def __init__(
        self,
        reader,
        writer,
        voice: PiperVoiceRKNN,
        info: Info,
        samples_per_chunk: int,
        auto_punctuation: str,
        default_speaker: Optional[str],
        length_scale: Optional[float],
        noise_scale: Optional[float],
        noise_w_scale: Optional[float],
        no_streaming: bool,
    ) -> None:
        super().__init__(reader, writer)
        self.voice = voice
        self.info_event = info.event()
        self.samples_per_chunk = samples_per_chunk
        self.auto_punctuation = auto_punctuation
        self.default_speaker = default_speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.no_streaming = no_streaming
        self.is_streaming = False
        self.stream_text = ""
        self.stream_voice = None

    async def handle_event(self, event: Event) -> bool:
        _LOGGER.debug("Received Wyoming event: %s", event.type)

        if Describe.is_type(event.type):
            await self.write_event(self.info_event)
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    return True

                synthesize = Synthesize.from_event(event)
                await self._handle_synthesize(synthesize)
                return True

            if self.no_streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.stream_text = ""
                self.stream_voice = start.voice
                return True

            if SynthesizeChunk.is_type(event.type):
                chunk = SynthesizeChunk.from_event(event)
                self.stream_text += chunk.text
                return True

            if SynthesizeStop.is_type(event.type):
                synthesize = Synthesize(text=self.stream_text, voice=self.stream_voice)
                await self._handle_synthesize(synthesize)
                await self.write_event(SynthesizeStopped().event())
                self.is_streaming = False
                self.stream_text = ""
                self.stream_voice = None
                return True
        except Exception as err:
            _LOGGER.exception("Synthesis failed")
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            return True

        return True

    async def _handle_synthesize(self, synthesize: Synthesize) -> None:
        text = normalize_text(synthesize.text, self.auto_punctuation)
        if not text:
            await self.write_event(AudioStart(rate=self.voice.config.sample_rate, width=2, channels=1).event())
            await self.write_event(AudioStop().event())
            return

        _LOGGER.info("Synthesizing text: %s", text)
        syn_config = self._make_synthesis_config(synthesize)
        wav_bytes = await asyncio.to_thread(synthesize_wav_bytes, self.voice, text, syn_config)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            rate = wav_file.getframerate()
            width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())

        bytes_per_sample = width * channels
        bytes_per_chunk = bytes_per_sample * self.samples_per_chunk
        num_chunks = int(math.ceil(len(audio_bytes) / bytes_per_chunk))
        for i in range(num_chunks):
            offset = i * bytes_per_chunk
            chunk = audio_bytes[offset : offset + bytes_per_chunk]
            await self.write_event(AudioChunk(audio=chunk, rate=rate, width=width, channels=channels).event())

        await self.write_event(AudioStop().event())

    def _make_synthesis_config(self, synthesize: Synthesize) -> SynthesisConfig:
        syn_config = SynthesisConfig()

        voice_speaker = self.default_speaker
        if synthesize.voice is not None and synthesize.voice.speaker:
            voice_speaker = synthesize.voice.speaker

        if voice_speaker:
            syn_config.speaker_id = self.voice.config.speaker_id_map.get(voice_speaker)
            if syn_config.speaker_id is None:
                try:
                    syn_config.speaker_id = int(voice_speaker)
                except ValueError:
                    _LOGGER.warning("No speaker '%s' for voice", voice_speaker)

        if self.length_scale is not None:
            syn_config.length_scale = self.length_scale
        if self.noise_scale is not None:
            syn_config.noise_scale = self.noise_scale
        if self.noise_w_scale is not None:
            syn_config.noise_w_scale = self.noise_w_scale

        return syn_config


def find_model_files(model_dir: Path) -> tuple[Path, Path, Path]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    encoder = next(iter(sorted(model_dir.glob("*.onnx"))), None)
    decoder = next(iter(sorted(model_dir.glob("*.rknn"))), None)
    config = model_dir / "config.json"

    if encoder is None:
        raise FileNotFoundError(f"Need encoder .onnx in model dir: {model_dir}")
    if decoder is None:
        raise FileNotFoundError(f"Need decoder .rknn in model dir: {model_dir}")
    if not config.is_file():
        raise FileNotFoundError(f"Need config.json in model dir: {model_dir}")

    return encoder, decoder, config


def get_decoder_chunk_size(decoder_rknn: RKNNLite) -> int:
    runtime = decoder_rknn.rknn_runtime
    try:
        attr = runtime.get_tensor_attr(0, is_output=False)
    except TypeError:
        attr = runtime.get_tensor_attr(0)

    dims = list(attr.dims)
    if len(dims) < 3:
        raise RuntimeError(f"Unexpected RKNN decoder input dims: {dims}")

    return int(dims[2])


def chunk_decoder_inputs(
    z: np.ndarray,
    y_mask: np.ndarray,
    chunk_size: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    if z.ndim != 3:
        raise ValueError(f"Expected z shape (1, channels, time), got {z.shape}")
    if y_mask.ndim != 3:
        raise ValueError(f"Expected y_mask shape (1, channels, time), got {y_mask.shape}")
    if z.shape[2] != y_mask.shape[2]:
        raise ValueError(f"z/y_mask time mismatch: {z.shape} vs {y_mask.shape}")
    if chunk_size <= 0:
        raise ValueError(f"Invalid decoder chunk size: {chunk_size}")

    chunks = []
    for start in range(0, z.shape[2], chunk_size):
        end = min(start + chunk_size, z.shape[2])
        real_size = end - start
        z_chunk = z[:, :, start:end]
        y_chunk = y_mask[:, :, start:end]

        if real_size < chunk_size:
            z_chunk = pad_time_axis(z_chunk, chunk_size)
            y_chunk = pad_time_axis(y_chunk, chunk_size)

        chunks.append((z_chunk, y_chunk, real_size))

    return chunks


def pad_time_axis(tensor: np.ndarray, target_size: int) -> np.ndarray:
    padded = np.zeros((tensor.shape[0], tensor.shape[1], target_size), dtype=tensor.dtype)
    padded[:, :, : tensor.shape[2]] = tensor
    return padded


def trim_decoder_output(audio: np.ndarray, real_size: int, chunk_size: int) -> np.ndarray:
    if real_size >= chunk_size:
        return audio

    flat = audio.reshape(-1)
    real_len = int(flat.shape[0] * real_size / chunk_size)
    trimmed = flat[:real_len]
    new_shape = list(audio.shape)
    new_shape[-1] = real_len
    return trimmed.reshape(new_shape)


def synthesize_wav_bytes(voice: PiperVoiceRKNN, text: str, syn_config: SynthesisConfig) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    return output.getvalue()


def normalize_text(text: str, auto_punctuation: str) -> str:
    text = " ".join(text.strip().splitlines())
    if text and auto_punctuation and text[-1] not in auto_punctuation:
        text += auto_punctuation[0]

    return text


def build_info(voice: PiperVoiceRKNN, voice_name: str, model_dir: Path, no_streaming: bool) -> Info:
    language = getattr(voice.config, "espeak_voice", None) or voice_name.split("_")[0].split("-")[0]
    speakers = None
    speaker_id_map = getattr(voice.config, "speaker_id_map", None)
    if speaker_id_map:
        speakers = [TtsVoiceSpeaker(name=speaker) for speaker in speaker_id_map]

    return Info(
        tts=[
            TtsProgram(
                name="rknpu-piper",
                description="Piper TTS with ONNX encoder and RKNN/RKNPU decoder",
                version=None,
                attribution=Attribution(name="rhasspy/piper", url="https://github.com/rhasspy/piper"),
                installed=True,
                voices=[
                    TtsVoice(
                        name=voice_name,
                        description=f"{voice_name} ({model_dir})",
                        version=None,
                        attribution=Attribution(name="rhasspy/piper", url="https://github.com/rhasspy/piper"),
                        installed=True,
                        languages=[language],
                        speakers=speakers,
                    )
                ],
                supports_synthesize_streaming=(not no_streaming),
            )
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--voice", default="piper-rknn")
    parser.add_argument("--speaker")
    parser.add_argument("--samples-per-chunk", type=int, default=DEFAULT_SAMPLES_PER_CHUNK)
    parser.add_argument("--auto-punctuation", default=".?!。？！．؟")
    parser.add_argument("--length-scale", type=float)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--noise-w-scale", "--noise-w", type=float)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    voice = PiperVoiceRKNN.load_from_dir(model_dir, use_cuda=args.use_cuda)
    info = build_info(voice, args.voice, model_dir, args.no_streaming)
    server = AsyncServer.from_uri(args.uri)

    _LOGGER.info("Starting Wyoming RKNN Piper server on %s", args.uri)

    try:
        await server.run(
            partial(
                RknpuPiperEventHandler,
                voice=voice,
                info=info,
                samples_per_chunk=args.samples_per_chunk,
                auto_punctuation=args.auto_punctuation,
                default_speaker=args.speaker,
                length_scale=args.length_scale,
                noise_scale=args.noise_scale,
                noise_w_scale=args.noise_w_scale,
                no_streaming=args.no_streaming,
            )
        )
    finally:
        voice.close()


if __name__ == "__main__":
    asyncio.run(main())
