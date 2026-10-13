# SPDX-License-Identifier: Apache-2.0
"""Mimi decode graphs for individual streams and batches of ready frames."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.architecture import AUDIO_CODEBOOKS_PER_STREAM
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.components.mimi_state import MimiDecodeState
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

logger = logging.getLogger(__name__)

# Note (wilsonzheng0327): Initialize library handles before graph capture.
GRAPH_WARMUP_STEPS = 2


@dataclass(kw_only=True)
class DecodeSlot:
    """One stream's state, backed by a pool row when index is set."""

    index: int | None
    decode_state: MimiDecodeState


@dataclass(kw_only=True)
class CapturedDecode:
    graph: ReplayableGraph
    input_codes: torch.Tensor
    output_waveform: torch.Tensor
    slot_indices: torch.Tensor


class MimiDecodeSlots:
    """Resident stream states with graphs selected by actual batch and chunk size."""

    def __init__(
        self, codec: MimiCodec, *, num_slots: int, max_graph_frames: int
    ) -> None:
        self.codec: MimiCodec = codec
        self.max_batch_size: int = max(num_slots, 1)
        self.state_pool: MimiDecodeState | None = None
        self.decode_states: list[MimiDecodeState] = []
        self.free_slots: list[int] = []
        self.graphs: dict[tuple[int, int], CapturedDecode] = {}
        self.slot_graphs: dict[tuple[int, int], CapturedDecode] = {}
        self.has_warned_full: bool = False
        device = codec.device
        graph_backend = (
            current_platform.get_device_graph_backend(device)
            if current_platform.enable_codec_decode_graph()
            else None
        )
        if graph_backend is None or max_graph_frames == 0 or num_slots == 0:
            logger.info(f"PersonaPlex Mimi streaming decode runs eager on {device}")
            return
        else:
            pass

        self.state_pool = codec.init_decode_state(batch_size=num_slots)
        self.decode_states = [
            self.state_pool.select_rows(slice(index, index + 1))
            for index in range(num_slots)
        ]
        self.free_slots = list(reversed(range(num_slots)))
        device_module = torch.get_device_module(device)
        capture_stream = device_module.Stream(device=device)
        graph_pool = graph_backend.graph_pool_handle()
        capture_started = time.perf_counter()
        allocated_bytes_before = device_module.memory_allocated(device)
        with (
            torch.inference_mode(),
            device_module.device(device),
            current_platform.graph_capture_attention(),
        ):
            capture_stream.wait_stream(device_module.current_stream(device))
            # Note (Codex): Capture largest batches first to reuse pool workspaces.
            for batch_size in range(num_slots, 0, -1):
                slot_indices = torch.arange(batch_size, device=device)
                for chunk_frames in range(max_graph_frames, 0, -1):
                    input_codes = torch.zeros(
                        (batch_size, AUDIO_CODEBOOKS_PER_STREAM, chunk_frames),
                        dtype=torch.long,
                        device=device,
                    )
                    with device_module.stream(capture_stream):
                        for _ in range(GRAPH_WARMUP_STEPS):
                            if batch_size == 1:
                                codec.decode_step(input_codes, self.decode_states[0])
                            else:
                                self.decode_selected(input_codes, slot_indices)
                    for slot_index in range(num_slots if batch_size == 1 else 1):
                        with graph_backend.capture(
                            pool=graph_pool,
                            stream=capture_stream,
                            thread_local_errors=True,
                        ) as graph:
                            if batch_size == 1:
                                output_waveform = codec.decode_step(
                                    input_codes, self.decode_states[slot_index]
                                )
                            else:
                                output_waveform = self.decode_selected(
                                    input_codes, slot_indices
                                )
                        captured_decode = CapturedDecode(
                            graph=graph,
                            input_codes=input_codes,
                            output_waveform=output_waveform,
                            slot_indices=slot_indices,
                        )
                        if batch_size == 1:
                            self.slot_graphs[(slot_index, chunk_frames)] = (
                                captured_decode
                            )
                        else:
                            self.graphs[(batch_size, chunk_frames)] = captured_decode
            device_module.current_stream(device).wait_stream(capture_stream)
            self.state_pool.reset()
            device_module.synchronize(device)
        graph_bytes = device_module.memory_allocated(device) - allocated_bytes_before
        logger.info(
            f"PersonaPlex Mimi decode graphs: {num_slots} stream slots, batches of "
            f"1-{num_slots}, chunks of 1-{max_graph_frames} frames in "
            f"{time.perf_counter() - capture_started:.1f}s, {graph_bytes / 2**20:.0f}MiB"
        )

    @torch.inference_mode()
    def acquire(self) -> DecodeSlot:
        """A fresh stream state, allocated eagerly when no resident slot is free."""
        if self.free_slots:
            slot_index = self.free_slots.pop()
            decode_state = self.decode_states[slot_index]
            decode_state.reset()
            return DecodeSlot(index=slot_index, decode_state=decode_state)
        elif self.decode_states and not self.has_warned_full:
            logger.warning(
                f"All {len(self.decode_states)} PersonaPlex Mimi decode slots are "
                "taken, so further streams decode eagerly; raise "
                "--code2wav.factory.num_decode_slots to at least the LM's "
                "max_running_requests"
            )
            self.has_warned_full = True
        else:
            pass
        return DecodeSlot(
            index=None, decode_state=self.codec.init_decode_state(batch_size=1)
        )

    def release(self, slot: DecodeSlot) -> None:
        if slot.index is not None:
            self.free_slots.append(slot.index)
        else:
            pass

    def decode_selected(
        self, codes_BKF: torch.Tensor, slot_indices: torch.Tensor
    ) -> torch.Tensor:
        assert self.state_pool is not None
        decode_state = self.state_pool.select_rows(slot_indices)
        waveform = self.codec.decode_step(codes_BKF, decode_state)
        self.state_pool.copy_rows(slot_indices, decode_state)
        return waveform

    @torch.inference_mode()
    def decode_step(self, codes_BKF: torch.Tensor, slot: DecodeSlot) -> torch.Tensor:
        """Decode one stream; the returned storage is valid until the next decode."""
        captured_decode = self.slot_graphs.get((slot.index, codes_BKF.shape[-1]))
        if captured_decode is None:
            return self.codec.decode_step(codes_BKF, slot.decode_state)
        else:
            captured_decode.input_codes.copy_(codes_BKF)
            captured_decode.graph.replay()
            return captured_decode.output_waveform

    @torch.inference_mode()
    def decode_batch(
        self, codes_BKF: torch.Tensor, slots: list[DecodeSlot]
    ) -> torch.Tensor:
        """Decode ready streams, with storage valid until the next decode."""
        batch_size = len(slots)
        if batch_size == 1:
            return self.decode_step(codes_BKF, slots[0])
        elif all(slot.index is not None for slot in slots):
            slot_indices = torch.tensor(
                [slot.index for slot in slots],
                dtype=torch.long,
                device=self.codec.device,
            )
            captured_decode = self.graphs.get((batch_size, codes_BKF.shape[-1]))
            if captured_decode is None:
                return self.decode_selected(codes_BKF, slot_indices)
            else:
                captured_decode.slot_indices.copy_(slot_indices)
                captured_decode.input_codes.copy_(codes_BKF)
                captured_decode.graph.replay()
                return captured_decode.output_waveform
        else:
            return torch.cat(
                [
                    self.codec.decode_step(codes[None], slot.decode_state)
                    for codes, slot in zip(codes_BKF, slots, strict=True)
                ]
            )


__all__ = ["DecodeSlot", "MimiDecodeSlots"]
