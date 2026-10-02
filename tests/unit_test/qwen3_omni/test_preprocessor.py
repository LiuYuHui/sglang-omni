# SPDX-License-Identifier: Apache-2.0
"""Check Qwen3-Omni transcription uploads and audio chat preprocessing."""

from __future__ import annotations

import asyncio
import io
import wave
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.proto.request import OmniRequest, StagePayload

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% for part in message['content'] %}"
    "{% if part['type'] == 'audio' %}"
    "{{ '<|audio_start|><|audio_pad|><|audio_end|>' }}"
    "{% else %}{{ part['text'] }}{% endif %}"
    "{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def wav_bytes(sample_rate_hz: int = 16000, sample_count: int = 4) -> bytes:
    samples = np.array([8192, -8192, 4096, -4096], dtype="<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate_hz)
        audio_file.writeframes(samples[:sample_count].tobytes())
    return buffer.getvalue()


@pytest.fixture
def audio_preprocessor() -> Qwen3OmniPreprocessor:
    template_processor = object.__new__(Qwen3OmniMoeProcessor)
    template_processor.tokenizer = PreTrainedTokenizerBase()
    template_processor.chat_template = CHAT_TEMPLATE
    preprocessor = object.__new__(Qwen3OmniPreprocessor)
    preprocessor.max_seq_len = None
    preprocessor.default_video_fps = None
    preprocessor.default_video_max_frames = None
    preprocessor.default_video_min_pixels = None
    preprocessor.default_video_max_pixels = None
    preprocessor.default_video_total_pixels = None
    preprocessor.processor = Mock(
        spec=Qwen3OmniMoeProcessor,
        return_value={
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "input_features": torch.zeros(1, 128, 4),
            "feature_attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
    )
    preprocessor.processor.apply_chat_template.side_effect = (
        template_processor.apply_chat_template
    )
    return preprocessor


@pytest.mark.parametrize("language", [None, "en", "zh", "fr"])
def test_transcription_upload_decodes_audio_and_sets_task(
    audio_preprocessor: Qwen3OmniPreprocessor, language: str | None
) -> None:
    payload = StagePayload(
        request_id="transcription-upload",
        request=OmniRequest(
            inputs={
                "audio_bytes": wav_bytes(),
                "filename": "sample.wav",
                "content_type": "audio/wav",
            },
            params={"task": "transcribe", "language": language, "max_new_tokens": 64},
            metadata={"task": "asr", "output_modalities": ["text"]},
        ),
        data=None,
    )

    result = asyncio.run(audio_preprocessor(payload))

    state = Qwen3OmniPipelineState.from_dict(result.data)
    assert state.prompt is not None
    prompt_text = state.prompt["prompt_text"]
    assert "transcribe" in prompt_text.lower()
    assert "original language" in prompt_text
    assert "only the transcription" in prompt_text
    assert prompt_text.count("<|audio_pad|>") == 1
    if language is not None:
        assert f"The spoken language is {language}." in prompt_text
    else:
        assert "The spoken language is" not in prompt_text
    processor_call = audio_preprocessor.processor.call_args
    assert len(processor_call.kwargs["audio"]) == 1
    np.testing.assert_array_equal(
        processor_call.kwargs["audio"][0],
        np.array([0.25, -0.25, 0.125, -0.125], dtype=np.float32),
    )
    assert "input_features" in state.encoder_inputs["audio_encoder"]
    assert state.encoder_inputs["image_encoder"] == {"_skip": True, "_result": {}}
    assert result.request.inputs is None
    assert result.request.params["max_new_tokens"] == 64
    assert result.request.metadata["output_modalities"] == ["text"]


@pytest.mark.parametrize("context", [None, "", "SGLang and Qwen3-Omni"])
def test_transcription_context_preserves_task_instruction(
    audio_preprocessor: Qwen3OmniPreprocessor, context: str | None
) -> None:
    payload = StagePayload(
        request_id="transcription-context",
        request=OmniRequest(
            inputs={"audio_bytes": wav_bytes()}, params={"prompt": context}
        ),
        data=None,
    )

    result = asyncio.run(audio_preprocessor(payload))

    state = Qwen3OmniPipelineState.from_dict(result.data)
    assert state.prompt is not None
    prompt_text = state.prompt["prompt_text"]
    assert "transcribe" in prompt_text.lower()
    assert "only the transcription" in prompt_text
    if context:
        assert f"Transcription context: {context}" in prompt_text
    else:
        assert "Transcription context:" not in prompt_text


def test_transcription_upload_resamples_audio(
    audio_preprocessor: Qwen3OmniPreprocessor,
) -> None:
    payload = StagePayload(
        request_id="transcription-resample",
        request=OmniRequest(inputs={"audio_bytes": wav_bytes(sample_rate_hz=8000)}),
        data=None,
    )

    asyncio.run(audio_preprocessor(payload))

    waveform = audio_preprocessor.processor.call_args.kwargs["audio"][0]
    assert waveform.dtype == np.float32
    assert waveform.shape == (8,)
    assert waveform[0] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "audio_bytes", [b"", b"invalid audio", "not bytes", wav_bytes(sample_count=0)]
)
def test_invalid_transcription_upload_reports_decode_error(
    audio_preprocessor: Qwen3OmniPreprocessor, audio_bytes: bytes | str
) -> None:
    payload = StagePayload(
        request_id="transcription-invalid",
        request=OmniRequest(inputs={"audio_bytes": audio_bytes}),
        data=None,
    )

    with pytest.raises(ValueError, match="could not decode the uploaded audio"):
        asyncio.run(audio_preprocessor(payload))
    audio_preprocessor.processor.assert_not_called()


def test_audio_chat_preserves_prompt_and_waveform(
    audio_preprocessor: Qwen3OmniPreprocessor,
) -> None:
    waveform = np.array([0.25, -0.25], dtype=np.float32)
    instruction = "Answer the question in the recording."
    payload = StagePayload(
        request_id="audio-chat",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": instruction}],
                "audios": [waveform],
            }
        ),
        data=None,
    )

    result = asyncio.run(audio_preprocessor(payload))

    state = Qwen3OmniPipelineState.from_dict(result.data)
    assert state.prompt is not None
    assert state.prompt["prompt_text"] == (
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{instruction}<|im_end|>\n<|im_start|>assistant\n"
    )
    assert audio_preprocessor.processor.call_args.kwargs["audio"][0] is waveform
