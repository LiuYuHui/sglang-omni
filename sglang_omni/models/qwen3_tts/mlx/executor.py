# SPDX-License-Identifier: Apache-2.0
"""Single-stage native MLX serving for Qwen3-TTS CustomVoice."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from sglang_omni.models.qwen3_tts.config import (
    load_qwen3_tts_checkpoint_config,
    normalize_qwen3_tts_model_type,
)
from sglang_omni.models.qwen3_tts.request_builders import (
    QWEN3_TTS_TASK_CUSTOM_VOICE,
    build_qwen3_tts_state,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

SUPPORTED_MLX_GENERATION_FIELDS = frozenset(
    {"max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty"}
)


def create_mlx_tts_executor(
    model_path: str,
    *,
    gpu_id: int | None = None,
    mlx_model_path: str | None = None,
    mlx_model_revision: str | None = None,
) -> SimpleScheduler:
    """Load a converted CustomVoice checkpoint and serve waveforms."""
    import mlx.core as mx
    from sglang.srt.hardware_backend.mlx.runtime import use_mlx

    from sglang_omni.models.qwen3_tts.mlx.generate import Qwen3TTSMlxGenerator

    if not current_platform.is_mps() or not use_mlx():
        raise RuntimeError("Qwen3-TTS MLX requires Apple Metal and SGLANG_USE_MLX=1")
    else:
        pass
    if gpu_id not in (None, 0):
        raise ValueError("Qwen3-TTS MLX supports only Metal device 0")
    else:
        pass
    if not mlx_model_path:
        raise ValueError("Qwen3-TTS MLX requires factory.mlx_model_path")
    else:
        pass
    checkpoint_config = load_qwen3_tts_checkpoint_config(model_path)
    model_type = normalize_qwen3_tts_model_type(checkpoint_config.get("tts_model_type"))
    if model_type != "custom_voice":
        raise ValueError(
            "Qwen3-TTS MLX currently supports CustomVoice checkpoints only"
        )
    else:
        pass

    converted_dir = Path(mlx_model_path).expanduser()
    if not converted_dir.is_dir():
        converted_dir = Path(
            snapshot_download(repo_id=mlx_model_path, revision=mlx_model_revision)
        )
    else:
        pass
    generator = Qwen3TTSMlxGenerator(converted_dir)
    speakers = {
        name.casefold(): name for name in generator.talker.artifact.talker_config.spk_id
    }

    def generate(payload: StagePayload) -> StagePayload:
        if payload.request.params.get("stream"):
            raise ValueError("Qwen3-TTS MLX currently supports stream=false only")
        else:
            pass
        tts_params = (payload.request.metadata or {}).get("tts_params")
        if not isinstance(tts_params, dict):
            tts_params = {}
        else:
            pass
        if float(tts_params.get("speed", 1.0)) != 1.0:
            raise ValueError("Qwen3-TTS MLX currently supports speed=1.0 only")
        else:
            pass
        state = build_qwen3_tts_state(payload, default_stream_codec_output=False)
        if state.task_type != QWEN3_TTS_TASK_CUSTOM_VOICE:
            raise ValueError("Qwen3-TTS MLX currently supports CustomVoice only")
        else:
            pass
        if state.instructions is not None:
            raise ValueError("Qwen3-TTS MLX currently does not support instructions")
        else:
            pass
        unsupported = state.generation_kwargs.keys() - SUPPORTED_MLX_GENERATION_FIELDS
        if unsupported:
            raise ValueError(
                f"Qwen3-TTS MLX does not support generation parameters: {sorted(unsupported)}"
            )
        else:
            pass
        voice = speakers.get((state.voice or "").casefold())
        if voice is None:
            raise ValueError(f"Qwen3-TTS MLX does not support voice {state.voice!r}")
        else:
            pass
        if state.seed is not None:
            mx.random.seed(state.seed)
        else:
            pass

        generation = state.generation_kwargs
        started = time.perf_counter()
        waveform, token_count = generator.generate(
            text=state.text,
            voice=voice,
            language=state.language,
            max_new_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation.get("temperature", 0.9)),
            top_k=int(generation.get("top_k", 50)),
            top_p=float(generation.get("top_p", 1.0)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
        )
        mx.eval(waveform)
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise RuntimeError("Qwen3-TTS MLX returned an empty or invalid waveform")
        else:
            pass
        state.completion_tokens = token_count
        state.engine_time_s = time.perf_counter() - started
        payload.data = audio_waveform_payload(
            audio,
            sample_rate=generator.decoder.output_sample_rate,
            modality="audio",
            source_hint="Qwen3-TTS MLX",
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        else:
            pass
        return payload

    return SimpleScheduler(generate)
