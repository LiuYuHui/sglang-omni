# SPDX-License-Identifier: Apache-2.0
"""The streaming codec stage renders per-request chunks into the whole-reply waveform."""

from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage


def decode_waveform(payload: dict) -> torch.Tensor:
    assert payload["sample_rate"] == SAMPLE_RATE
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def new_scheduler(codec: MimiCodec) -> PersonaPlexCode2WavScheduler:
    return PersonaPlexCode2WavScheduler(
        MimiDecodeSlots(codec, num_slots=8, max_graph_frames=1),
        compute_fn=lambda payload: payload,
    )


def start_stream(
    scheduler: PersonaPlexCode2WavScheduler, request_id: str
) -> StagePayload:
    payload = StagePayload(
        request_id, request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    scheduler.stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)
    return payload


def chunk_message(
    request_id: str, codes_FK: torch.Tensor, *, num_samples: int = 0
) -> IncomingMessage:
    return IncomingMessage(
        request_id,
        "stream_chunk",
        StreamItem(
            chunk_id=0,
            data=codes_FK,
            from_stage="lm",
            metadata={"num_samples": num_samples},
        ),
    )


def drain_outputs(scheduler: PersonaPlexCode2WavScheduler) -> list[OutgoingMessage]:
    return [scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())]


def test_interleaved_requests_stream_their_own_waveforms(random_codec):
    codec = random_codec
    scheduler = new_scheduler(codec)
    codes = {
        "a": torch.randint(0, 2048, (4, 8), generator=torch.Generator().manual_seed(1)),
        "b": torch.randint(0, 2048, (4, 8), generator=torch.Generator().manual_seed(2)),
    }
    whole = {rid: codec.decode(c.T[None])[0, 0] for rid, c in codes.items()}
    payloads = {rid: start_stream(scheduler, rid) for rid in codes}

    streamed = {rid: [] for rid in codes}

    def push(rid: str, chunk: torch.Tensor) -> None:
        (message,) = scheduler.on_stream_chunk(
            rid, SimpleNamespace(data=chunk, metadata=None)
        )
        assert message.type == "stream"
        streamed[rid].append(decode_waveform(message.data))

    for frame in range(4):
        push("a", codes["a"][frame : frame + 1])
        if frame % 2 == 1:
            push("b", codes["b"][frame - 1 : frame + 1])

    for rid in codes:
        torch.testing.assert_close(
            torch.cat(streamed[rid]), whole[rid], atol=1e-5, rtol=1e-5
        )
        (result,) = scheduler.on_stream_done(rid)
        assert result.type == "result"
        assert result.data.request is payloads[rid].request
        torch.testing.assert_close(
            decode_waveform(result.data.data), whole[rid], atol=1e-5, rtol=1e-5
        )


def test_a_reply_with_frames_streams_whatever_arrives_first(random_codec):
    scheduler = new_scheduler(random_codec)
    empty = StagePayload(
        "a", request=OmniRequest(inputs={}), data=PersonaPlexState().to_dict()
    )
    assert not scheduler.is_streaming_payload(empty)
    with_frames = StagePayload(
        "a",
        request=OmniRequest(inputs={}),
        data=PersonaPlexState(codes=torch.zeros(3, 8, dtype=torch.long)).to_dict(),
    )
    assert scheduler.is_streaming_payload(with_frames)


def test_abort_clears_stream_state(random_codec):
    scheduler = new_scheduler(random_codec)
    start_stream(scheduler, "a")
    assert scheduler.on_stream_done("a") != []
    scheduler.clear_stream_state("a")
    assert scheduler.on_stream_done("a") == []
    assert scheduler.on_stream_done("never-started") == []


@pytest.mark.parametrize(
    "num_samples",
    [0, 4 * SAMPLES_PER_FRAME, 3 * SAMPLES_PER_FRAME + 100],
    ids=["unspecified", "whole", "partial"],
)
def test_reply_length_matches_the_caller_in_chunks_and_final_payload(
    random_codec: MimiCodec, num_samples: int
) -> None:
    codec = random_codec
    scheduler = new_scheduler(codec)
    frames, samples_per_frame = 4, codec.samples_per_frame
    expected_samples = num_samples or frames * samples_per_frame
    codes = torch.randint(
        0, 2048, (frames, 8), generator=torch.Generator().manual_seed(3)
    )
    whole = codec.decode(codes.T[None])[0, 0]
    assert whole.shape[-1] == frames * samples_per_frame

    streamed = []
    for frame in range(frames):
        (message,) = scheduler.on_stream_chunk(
            "a",
            SimpleNamespace(
                data=codes[frame : frame + 1], metadata={"num_samples": num_samples}
            ),
        )
        streamed.append(decode_waveform(message.data))
    torch.testing.assert_close(
        torch.cat(streamed), whole[:expected_samples], atol=1e-5, rtol=1e-5
    )
    assert streamed[-1].shape[-1] == expected_samples - 3 * samples_per_frame

    # Note (wilsonzheng0327): The terminal payload lands after every chunk, as it does
    # when the LM stage finishes.
    start_stream(scheduler, "a")
    (result,) = scheduler.on_stream_done("a")
    reply = decode_waveform(result.data.data)
    torch.testing.assert_close(reply, whole[:expected_samples], atol=1e-5, rtol=1e-5)


def test_ready_chunks_decode_together_and_a_single_ready_chunk_emits_immediately(
    small_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(small_codec)
    codes = torch.arange(8).expand(8, 8) % small_codec.spec.codebook_size
    messages = [chunk_message(str(i), codes[i : i + 1]) for i in range(8)]
    for message in messages[1:]:
        scheduler.inbox.put(message)
    with patch.object(
        small_codec, "decode_step", wraps=small_codec.decode_step
    ) as decode:
        scheduler.handle_message(messages[0], None)
        assert [call.args[0].shape[0] for call in decode.call_args_list] == [8]
    assert [message.request_id for message in drain_outputs(scheduler)] == list(
        map(str, range(8))
    )
    scheduler.handle_message(chunk_message("0", codes[:1]), None)
    assert [message.request_id for message in drain_outputs(scheduler)] == ["0"]


def test_batched_streams_keep_independent_lengths_and_final_waveforms(
    small_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(small_codec)
    codes = torch.randint(
        small_codec.spec.codebook_size,
        (2, 3, 8),
        generator=torch.Generator().manual_seed(14),
    )
    expected = [small_codec.decode(stream.T[None])[0, 0] for stream in codes]
    for request_id in ("a", "b"):
        start_stream(scheduler, request_id)
    trim_samples = small_codec.samples_per_frame + 3
    for frame in range(3):
        scheduler.inbox.put(chunk_message("b", codes[1, frame : frame + 1]))
        scheduler.handle_message(
            chunk_message("a", codes[0, frame : frame + 1], num_samples=trim_samples),
            None,
        )
    streamed = drain_outputs(scheduler)
    for stream_index, request_id in enumerate(("a", "b")):
        parts = [
            decode_waveform(message.data)
            for message in streamed
            if message.request_id == request_id
        ]
        length = trim_samples if request_id == "a" else expected[stream_index].numel()
        expected_waveform = expected[stream_index][:length]
        torch.testing.assert_close(
            torch.cat(parts), expected_waveform, atol=1e-5, rtol=1e-5
        )
        scheduler.handle_message(IncomingMessage(request_id, "stream_done"), None)
        (finished,) = drain_outputs(scheduler)
        assert finished.type == "result"
        torch.testing.assert_close(
            decode_waveform(finished.data.data), expected_waveform, atol=1e-5, rtol=1e-5
        )


def test_stream_batch_failure_aborts_its_group_and_later_width_still_decodes(
    small_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(small_codec)
    valid_codes = torch.zeros(2, 8, dtype=torch.long)
    bad_codes = torch.full((1, 8), small_codec.spec.codebook_size, dtype=torch.long)
    scheduler.inbox.put(chunk_message("bad", bad_codes))
    scheduler.inbox.put(chunk_message("later", valid_codes))
    scheduler.handle_message(chunk_message("affected", valid_codes[:1]), None)
    results = drain_outputs(scheduler)
    assert [(message.request_id, message.type) for message in results] == [
        ("affected", "error"),
        ("bad", "error"),
        ("later", "stream"),
    ]
    assert scheduler.is_aborted("affected") and scheduler.is_aborted("bad")
    assert not scheduler.is_aborted("later")


def test_abort_during_cpu_copy_preserves_peer_and_reused_slot(
    small_codec: MimiCodec,
) -> None:
    scheduler = new_scheduler(small_codec)
    codes = torch.randint(
        small_codec.spec.codebook_size,
        (2, 2, small_codec.spec.num_codebooks),
        generator=torch.Generator().manual_seed(15),
    )
    expected = [small_codec.decode(stream.T[None])[0, 0] for stream in codes]
    for request_id in ("aborted", "peer"):
        start_stream(scheduler, request_id)
    copy_entered, abort_marked, abort_finished = Event(), Event(), Event()
    record_abort = scheduler.record_aborted_request_id
    copy_to_cpu = torch.Tensor.cpu

    def mark_abort(request_id: str) -> None:
        record_abort(request_id)
        abort_marked.set()

    def copy_after_abort(waveforms: torch.Tensor) -> torch.Tensor:
        copy_entered.set()
        assert abort_marked.wait(timeout=5)
        assert not abort_finished.is_set()
        return copy_to_cpu(waveforms)

    def abort_request() -> None:
        if copy_entered.wait(timeout=5):
            scheduler.abort("aborted")
            abort_finished.set()
        else:
            pass

    worker = Thread(target=abort_request, daemon=True)
    scheduler.inbox.put(chunk_message("peer", codes[1, :1]))
    with (
        patch.object(scheduler, "record_aborted_request_id", side_effect=mark_abort),
        patch.object(torch.Tensor, "cpu", new=copy_after_abort),
    ):
        worker.start()
        try:
            scheduler.handle_message(chunk_message("aborted", codes[0, :1]), None)
        finally:
            copy_entered.set()
            worker.join(timeout=5)
        assert not worker.is_alive() and abort_finished.is_set()
    (peer,) = drain_outputs(scheduler)
    assert peer.request_id == "peer" and peer.type == "stream"
    torch.testing.assert_close(
        decode_waveform(peer.data),
        expected[1][: small_codec.samples_per_frame],
        atol=1e-6,
        rtol=1e-5,
    )
    scheduler.inbox.put(chunk_message("fresh", codes[0, :1]))
    scheduler.handle_message(chunk_message("peer", codes[1, 1:]), None)
    peer, fresh = drain_outputs(scheduler)
    assert [peer.request_id, fresh.request_id] == ["peer", "fresh"]
    torch.testing.assert_close(
        decode_waveform(peer.data),
        expected[1][small_codec.samples_per_frame :],
        atol=1e-6,
        rtol=1e-5,
    )
    expected_fresh = small_codec.decode(codes[0, :1].T[None])[0, 0]
    torch.testing.assert_close(
        decode_waveform(fresh.data), expected_fresh, atol=1e-6, rtol=1e-5
    )
