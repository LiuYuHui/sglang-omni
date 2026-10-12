# SPDX-License-Identifier: Apache-2.0
"""Streams the agent's Mimi codes into audio as they are generated.

Mimi's decoder is causal and its chunked path lands on exactly the samples a
whole-sequence decode would, so each frame goes out as soon as its codes
arrive: 1920 samples per 80 ms frame, no holdback.
"""

from __future__ import annotations

from itertools import groupby
from typing import Protocol

import torch

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE
from sglang_omni.models.personaplex.mimi_decode_graph import DecodeSlot, MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

SOURCE_HINT = "PersonaPlex"


def trim_to_caller(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Cut the reply back to the caller recording's own length.

    Frames are whole 80 ms, so a recording that is not a multiple of one is
    padded before encoding; the reference trims that padding off the reply.
    """
    if num_samples and waveform.shape[-1] > num_samples:
        return waveform[..., :num_samples]
    else:
        pass
    return waveform


class AudioPayloadDecoder(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload: ...


class StreamState:
    def __init__(self, decode_slot: DecodeSlot) -> None:
        self.decode_slot = decode_slot
        self.audio_parts: list[torch.Tensor] = []
        self.emitted = 0


class PersonaPlexCode2WavScheduler(StreamingSimpleScheduler):
    can_batch_stream_chunks: bool = True
    stream_chunk_batch_distinct_requests: bool = True

    def __init__(
        self, decode_slots: MimiDecodeSlots, *, compute_fn: AudioPayloadDecoder
    ) -> None:
        super().__init__(compute_fn)
        self.decode_slots = decode_slots
        self.stream_chunk_batch_max: int = decode_slots.max_batch_size
        self.stream_states: dict[str, StreamState] = {}

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        # Note (wilsonzheng0327): The LM streams every frame it produces, so a reply
        # with frames has chunks on the way whichever channel lands first; only an
        # empty reply is rendered whole.
        codes = PersonaPlexState.from_dict(payload.data).codes
        return codes is not None and codes.shape[0] > 0

    def stream_state(self, request_id: str) -> StreamState:
        state = self.stream_states.get(request_id)
        if state is None:
            state = StreamState(self.decode_slots.acquire())
            self.stream_states[request_id] = state
        else:
            pass
        return state

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.stream_state(request_id)

    def clear_stream_state(self, request_id: str) -> None:
        state = self.stream_states.pop(request_id, None)
        if state is not None:
            self.decode_slots.release(state.decode_slot)
        else:
            pass

    @torch.inference_mode()
    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        codes = self.codes_from_chunk(item).to(device=self.decode_slots.codec.device)
        state = self.stream_state(request_id)
        waveform = self.decode_slots.decode_step(codes.T[None], state.decode_slot)
        waveform = waveform[0, 0].float().cpu()
        return [self.waveform_message(request_id, state, item, waveform)]

    @torch.inference_mode()
    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        with self.state_lock:
            chunks = [
                (request_id, item, self.codes_from_chunk(item))
                for request_id, item in items
            ]
            for _, group in groupby(
                chunks, key=lambda chunk: (chunk[2].shape[0], chunk[2].device)
            ):
                chunks = [chunk for chunk in group if not self.is_aborted(chunk[0])]
                if not chunks:
                    continue
                else:
                    pass
                try:
                    states = [
                        self.stream_state(request_id) for request_id, _, _ in chunks
                    ]
                    codes = torch.stack([codes.T for _, _, codes in chunks]).to(
                        device=self.decode_slots.codec.device
                    )
                    waveforms = (
                        self.decode_slots.decode_batch(
                            codes, [state.decode_slot for state in states]
                        )
                        .float()
                        .cpu()
                    )
                except Exception as exc:
                    for request_id, _, _ in chunks:
                        if not self.is_aborted(request_id):
                            self.emit_error(request_id, exc)
                            self.abort(request_id)
                        else:
                            pass
                    continue
                for (request_id, item, _), state, waveform in zip(
                    chunks, states, waveforms[:, 0], strict=True
                ):
                    message = self.waveform_message(request_id, state, item, waveform)
                    if not self.is_aborted(request_id):
                        self.outbox.put(message)
                    else:
                        pass

    def codes_from_chunk(self, item: StreamItem) -> torch.Tensor:
        assert isinstance(item.data, torch.Tensor)
        return item.data.to(dtype=torch.long)

    def waveform_message(
        self,
        request_id: str,
        state: StreamState,
        item: StreamItem,
        waveform: torch.Tensor,
    ) -> OutgoingMessage:
        # Note (wilsonzheng0327): The terminal payload only arrives after the LM finishes,
        # so the caller length travels with each chunk.
        num_samples = int((item.metadata or {}).get("num_samples") or 0)
        if num_samples:
            waveform = waveform[..., : max(num_samples - state.emitted, 0)]
        else:
            pass
        state.emitted += waveform.shape[-1]
        state.audio_parts.append(waveform)
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=audio_waveform_payload(
                waveform,
                sample_rate=SAMPLE_RATE,
                modality="audio",
                source_hint=SOURCE_HINT,
            ),
            metadata={"modality": "audio"},
        )

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states.get(request_id)
        if state is None:
            return []
        else:
            pass
        waveform = torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=request_id,
                    request=self.stream_payloads[request_id].request,
                    data=audio_waveform_payload(
                        waveform,
                        sample_rate=SAMPLE_RATE,
                        modality="audio",
                        source_hint=SOURCE_HINT,
                    ),
                ),
            )
        ]


__all__ = ["PersonaPlexCode2WavScheduler", "trim_to_caller"]
