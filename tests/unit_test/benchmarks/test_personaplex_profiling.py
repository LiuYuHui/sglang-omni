# SPDX-License-Identifier: Apache-2.0
"""Validate PersonaPlex benchmark request measurements without a model."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from typing import Literal

import numpy as np
import pytest

from benchmarks.eval import personaplex_profiling_workload
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.pipeline.coordinator import Coordinator


@pytest.fixture
def client() -> Client:
    coordinator = Coordinator(
        completion_endpoint="inproc://profiling-test-completion",
        abort_endpoint="inproc://profiling-test-abort",
        entry_stage="preprocessing",
    )
    return Client(coordinator)


@pytest.mark.asyncio
async def test_stream_and_terminal_audio_are_measured_once(
    monkeypatch: pytest.MonkeyPatch, client: Client
) -> None:
    elapsed_seconds = 10.0
    waveform = np.array([0.0, 0.125, -0.25, 0.5], dtype=np.float32)

    def clock_seconds() -> float:
        return elapsed_seconds

    async def generate(
        self: Client, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        nonlocal elapsed_seconds
        assert self is client and request.stream and request_id == "measured"
        elapsed_seconds = 10.1
        yield GenerateChunk(request_id=request_id, text="partial")
        elapsed_seconds = 10.2
        yield GenerateChunk(
            request_id=request_id,
            audio_data=waveform[:0],
            sample_rate=4,
        )
        elapsed_seconds = 10.25
        yield GenerateChunk(
            request_id=request_id,
            audio_data=waveform[:2],
            sample_rate=4,
        )
        elapsed_seconds = 10.5
        yield GenerateChunk(
            request_id=request_id,
            audio_data=waveform[2:],
            sample_rate=4,
        )
        elapsed_seconds = 10.75
        yield GenerateChunk(
            request_id=request_id,
            audio_data=waveform,
            sample_rate=4,
            finish_reason="stop",
        )
        elapsed_seconds = 10.9
        yield GenerateChunk(
            request_id=request_id,
            text="complete transcript",
            finish_reason="stop",
        )
        elapsed_seconds = 11.0

    monkeypatch.setattr(Client, "generate", generate)
    monkeypatch.setattr(
        personaplex_profiling_workload.time, "perf_counter", clock_seconds
    )

    measurement = await personaplex_profiling_workload.measure_request(
        client, GenerateRequest(stream=True), "measured", 1.0
    )

    assert measurement.output_samples == 4
    assert measurement.stream_chunk_count == 3
    assert measurement.sample_rate == 4
    assert measurement.audio_duration_seconds == 1.0
    assert measurement.time_to_first_audio_milliseconds == pytest.approx(250.0)
    assert measurement.latency_milliseconds == pytest.approx(1000.0)
    assert measurement.real_time_factor == pytest.approx(1.0)
    assert measurement.text == "complete transcript"
    assert measurement.audio_sha256 == hashlib.sha256(waveform.tobytes()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario, expected_error",
    [
        ("missing_terminal", "terminal waveform"),
        ("empty_terminal", "no audio samples"),
        ("different_stream", "Streamed audio differs"),
        ("different_sample_rate", "Sample rate changed"),
        ("invalid_sample_rate", "positive sample rate"),
    ],
)
async def test_invalid_audio_cannot_produce_a_successful_measurement(
    monkeypatch: pytest.MonkeyPatch,
    client: Client,
    scenario: Literal[
        "missing_terminal",
        "empty_terminal",
        "different_stream",
        "different_sample_rate",
        "invalid_sample_rate",
    ],
    expected_error: str,
) -> None:
    waveform = np.array([0.25, -0.25], dtype=np.float32)

    async def generate(
        self: Client, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        assert self is client and request.stream and request_id is not None
        if scenario == "empty_terminal":
            yield GenerateChunk(
                request_id=request_id,
                audio_data=waveform[:0],
                sample_rate=4,
                finish_reason="stop",
            )
        else:
            yield GenerateChunk(
                request_id=request_id,
                audio_data=waveform,
                sample_rate=0 if scenario == "invalid_sample_rate" else 4,
            )
            if scenario == "missing_terminal":
                return
            else:
                yield GenerateChunk(
                    request_id=request_id,
                    audio_data=(
                        -waveform if scenario == "different_stream" else waveform
                    ),
                    sample_rate=8 if scenario == "different_sample_rate" else 4,
                    finish_reason="stop",
                )

    monkeypatch.setattr(Client, "generate", generate)

    with pytest.raises(ValueError, match=expected_error):
        await personaplex_profiling_workload.measure_request(
            client, GenerateRequest(stream=True), scenario, 1.0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("text_finishes_first", [False, True])
async def test_empty_terminal_transcript_remains_empty(
    monkeypatch: pytest.MonkeyPatch, client: Client, text_finishes_first: bool
) -> None:
    waveform = np.array([0.0, 0.25], dtype=np.float32)

    async def generate(
        self: Client, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        assert self is client and request.stream and request_id is not None
        yield GenerateChunk(request_id=request_id, text="partial text")
        terminal_text = GenerateChunk(
            request_id=request_id, text="", finish_reason="stop"
        )
        if text_finishes_first:
            yield terminal_text
        else:
            pass
        yield GenerateChunk(
            request_id=request_id,
            audio_data=waveform,
            sample_rate=2,
            finish_reason="stop",
        )
        if not text_finishes_first:
            yield terminal_text
        else:
            pass

    monkeypatch.setattr(Client, "generate", generate)

    measurement = await personaplex_profiling_workload.measure_request(
        client, GenerateRequest(stream=True), "empty-transcript", 1.0
    )

    assert measurement.text == ""


@pytest.mark.asyncio
async def test_audio_hash_preserves_float32_changes_below_pcm_resolution(
    monkeypatch: pytest.MonkeyPatch, client: Client
) -> None:
    original = np.array([0.0001], dtype=np.float32)
    changed = np.nextafter(original, np.float32(1.0))
    assert (original * 32767).astype(np.int16).tobytes() == (changed * 32767).astype(
        np.int16
    ).tobytes()

    async def generate(
        self: Client, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        assert self is client and request.stream and request_id is not None
        yield GenerateChunk(
            request_id=request_id,
            audio_data=changed if request_id == "changed" else original,
            sample_rate=24000,
            finish_reason="stop",
        )

    monkeypatch.setattr(Client, "generate", generate)
    request = GenerateRequest(stream=True)
    baseline = await personaplex_profiling_workload.measure_request(
        client, request, "baseline", 1.0
    )
    profiled = await personaplex_profiling_workload.measure_request(
        client, request, "profiled", 1.0
    )
    changed_measurement = await personaplex_profiling_workload.measure_request(
        client, request, "changed", 1.0
    )

    assert baseline.audio_sha256 == profiled.audio_sha256
    assert baseline.audio_sha256 != changed_measurement.audio_sha256


@pytest.mark.asyncio
async def test_invalid_audio_closes_active_generation(
    monkeypatch: pytest.MonkeyPatch, client: Client
) -> None:
    generator_closed = False

    async def generate(
        self: Client, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        nonlocal generator_closed
        assert self is client and request.stream and request_id is not None
        try:
            yield GenerateChunk(
                request_id=request_id,
                audio_data=b"encoded audio",
                sample_rate=24000,
            )
            await asyncio.Future[None]()
        finally:
            generator_closed = True

    monkeypatch.setattr(Client, "generate", generate)

    with pytest.raises(TypeError, match="NumPy waveform"):
        await personaplex_profiling_workload.measure_request(
            client, GenerateRequest(stream=True), "invalid-audio", 1.0
        )

    assert generator_closed


@pytest.mark.asyncio
async def test_request_timeout_cancels_generation(
    monkeypatch: pytest.MonkeyPatch, client: Client
) -> None:
    generator_closed = False

    async def generate(
        self: Client, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        nonlocal generator_closed
        assert self is client and request.stream and request_id is not None
        try:
            await asyncio.Future[None]()
            yield GenerateChunk(request_id=request_id)
        finally:
            generator_closed = True

    monkeypatch.setattr(Client, "generate", generate)

    with pytest.raises(asyncio.TimeoutError):
        await personaplex_profiling_workload.measure_request(
            client, GenerateRequest(stream=True), "timeout", 0.01
        )

    assert generator_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_pass_limits_concurrency_and_preserves_request_order(
    monkeypatch: pytest.MonkeyPatch, client: Client, concurrency: int
) -> None:
    active_requests = 0
    peak_active_requests = 0
    started_request_ids: list[str] = []
    waveform = np.array([0.0, 0.25], dtype=np.float32)

    async def generate(
        self: Client, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        nonlocal active_requests, peak_active_requests
        assert self is client and request.stream and request_id is not None
        active_requests += 1
        peak_active_requests = max(peak_active_requests, active_requests)
        started_request_ids.append(request_id)
        try:
            await asyncio.sleep(0)
            yield GenerateChunk(
                request_id=request_id,
                audio_data=waveform,
                sample_rate=2,
                finish_reason="stop",
            )
        finally:
            active_requests -= 1

    monkeypatch.setattr(Client, "generate", generate)

    measurement = await personaplex_profiling_workload.measure_pass(
        client, GenerateRequest(stream=True), "measured", 5, concurrency, 1.0
    )

    assert peak_active_requests == concurrency
    assert active_requests == 0
    assert started_request_ids == [f"measured-{index}" for index in range(5)]
    assert [
        request.request_id for request in measurement.requests
    ] == started_request_ids
    assert measurement.audio_seconds_per_wall_second == pytest.approx(
        5 / measurement.wall_seconds
    )


@pytest.mark.asyncio
async def test_failed_pass_cancels_and_awaits_sibling_requests(
    monkeypatch: pytest.MonkeyPatch, client: Client
) -> None:
    sibling_started = asyncio.Event()
    started_request_ids: set[str] = set()
    cancelled_request_ids: set[str] = set()
    finished_request_ids: set[str] = set()

    async def generate(
        self: Client, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        assert self is client and request.stream and request_id is not None
        started_request_ids.add(request_id)
        try:
            if request_id == "failed-0":
                await sibling_started.wait()
                raise RuntimeError("generation failed")
            else:
                sibling_started.set()
                await asyncio.Future[None]()
                yield GenerateChunk(request_id=request_id)
        except asyncio.CancelledError:
            cancelled_request_ids.add(request_id)
            raise
        finally:
            finished_request_ids.add(request_id)

    monkeypatch.setattr(Client, "generate", generate)

    with pytest.raises(RuntimeError, match="generation failed"):
        await personaplex_profiling_workload.measure_pass(
            client, GenerateRequest(stream=True), "failed", 3, 2, 1.0
        )

    assert "failed-1" in started_request_ids
    assert finished_request_ids == started_request_ids
    assert cancelled_request_ids == started_request_ids - {"failed-0"}
