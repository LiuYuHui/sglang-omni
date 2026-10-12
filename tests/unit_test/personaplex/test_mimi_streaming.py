# SPDX-License-Identifier: Apache-2.0
"""Mimi streaming preserves waveforms and independent stream state."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.personaplex.mimi_decode_graph as mimi_decode_graph
from sglang_omni.models.personaplex.architecture import MIMI
from sglang_omni.models.personaplex.components.causal_conv import (
    CausalConv1d,
    CausalConvTranspose1d,
)
from sglang_omni.models.personaplex.components.mimi import MimiCodec, rename_mimi_key
from sglang_omni.models.personaplex.components.mimi_state import MimiDecodeState
from sglang_omni.models.personaplex.components.mimi_transformer import (
    MimiAttention,
    MimiTransformer,
)
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.platforms import current_platform


def test_causal_conv_chunks_match_whole():
    torch.manual_seed(1)
    for kernel, stride, dilation, mode in (
        (7, 1, 1, "constant"),
        (8, 4, 1, "constant"),
        (3, 1, 2, "constant"),
        (4, 2, 1, "replicate"),
    ):
        conv = CausalConv1d(
            3, 5, kernel, stride=stride, dilation=dilation, pad_mode=mode
        )
        x = torch.randn(2, 3, 48)
        whole = conv(x)
        state = conv.init_state(batch_size=2)
        chunks = [conv.step(x[..., i : i + 8], state) for i in range(0, 48, 8)]
        torch.testing.assert_close(torch.cat(chunks, -1), whole, atol=1e-6, rtol=1e-5)


def test_causal_conv_transpose_chunks_match_whole():
    torch.manual_seed(2)
    for kernel, stride, groups in ((16, 8, 1), (4, 2, 6)):
        convtr = CausalConvTranspose1d(6, 6, kernel, stride=stride, groups=groups)
        x = torch.randn(1, 6, 10)
        whole = convtr(x)
        state = convtr.init_state(batch_size=1)
        chunks = [convtr.step(x[..., i : i + 1], state) for i in range(10)]
        torch.testing.assert_close(torch.cat(chunks, -1), whole, atol=1e-6, rtol=1e-5)


def test_codec_encode_and_decode_step_match_whole(random_codec):
    codec = random_codec
    frames = 5
    x = torch.randn(1, 1, codec.samples_per_frame * frames)
    codes = codec.encode(x)
    assert codes.shape == (1, 8, frames)
    state = codec.init_encode_state(batch_size=1)
    step = codec.samples_per_frame
    chunked = torch.cat(
        [
            codec.encode_step(x[..., i : i + step], state)
            for i in range(0, x.shape[-1], step)
        ],
        -1,
    )
    assert torch.equal(chunked, codes)

    whole = codec.decode(codes)
    assert whole.shape == (1, 1, codec.samples_per_frame * frames)
    state = codec.init_decode_state(batch_size=1)
    chunked = torch.cat(
        [codec.decode_step(codes[..., f : f + 1], state) for f in range(frames)], -1
    )
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-5)


def test_a_reset_decode_state_decodes_like_a_fresh_one(random_codec):
    codec = random_codec
    generator = torch.Generator().manual_seed(4)
    earlier, codes = torch.randint(0, 2048, (2, 1, 8, 3), generator=generator)

    def stream(codes_BKF: torch.Tensor, state: MimiDecodeState) -> torch.Tensor:
        return torch.cat(
            [codec.decode_step(codes_BKF[..., f : f + 1], state) for f in range(3)],
            -1,
        )

    fresh = stream(codes, codec.init_decode_state(batch_size=1))
    state = codec.init_decode_state(batch_size=1)
    stream(earlier, state)
    state.reset()
    assert torch.equal(stream(codes, state), fresh)


def test_checkpoint_names_map_onto_the_module_tree():
    codec = MimiCodec()
    expected = set(codec.state_dict())
    checkpoint_names = [
        "encoder.model.0.conv.conv.weight",
        "encoder.model.1.block.1.conv.conv.bias",
        "decoder.model.2.convtr.convtr.weight",
        "downsample.conv.conv.conv.weight",
        "upsample.convtr.convtr.convtr.weight",
        "encoder_transformer.transformer.layers.0.self_attn.in_proj_weight",
        "decoder_transformer.transformer.layers.7.layer_scale_2.scale",
        "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        "quantizer.rvq_rest.vq.layers.6._codebook.cluster_usage",
    ]
    for name in checkpoint_names:
        assert rename_mimi_key(name) in expected, name
    assert (
        rename_mimi_key("quantizer.rvq_rest.vq.layers.7._codebook.embedding_sum")
        is None
    )


SMALL = replace(MIMI, context=6, num_layers=2, dim=16, num_heads=2, ffn_dim=8)


def small_transformer() -> MimiTransformer:
    torch.manual_seed(4)
    transformer = MimiTransformer(SMALL).eval()
    with torch.no_grad():
        for parameter in transformer.parameters():
            parameter.normal_(std=0.2)
    return transformer


def influenced_steps(chunk: int) -> list[int]:
    """Which steps still depend on step 0, feeding chunk steps at a time."""
    torch.manual_seed(0)
    attention = MimiAttention(
        dim=8, num_heads=2, context=SMALL.context, max_period=1e4, write_chunk=chunk
    )
    with torch.no_grad():
        attention.in_proj_weight.normal_()
        attention.out_proj.weight.normal_()
    x = torch.randn(1, 4 * SMALL.context, 8)
    changed = x.clone()
    changed[:, 0] += 1.0

    def run(inp):
        state = attention.init_state(batch_size=1)
        with torch.no_grad():
            parts = [
                attention(inp[:, t : t + chunk], offset=t, state=state)
                for t in range(0, inp.shape[1], chunk)
            ]
        return torch.cat(parts, 1)

    differs = (run(x) - run(changed)).abs().amax(dim=(0, 2)) > 1e-6
    return differs.nonzero().flatten().tolist()


def test_the_ring_drops_its_oldest_step_as_the_reference_does():
    """The reference labels the slot it is about to overwrite as a future position,
    so once the ring is full its oldest entry leaves the window."""
    assert influenced_steps(1) == list(range(SMALL.context - 1))
    assert influenced_steps(SMALL.frame_ratio) == list(
        range(SMALL.context - SMALL.frame_ratio)
    )


@pytest.mark.parametrize(
    "length", [SMALL.context - 1, SMALL.context, 3 * SMALL.context + 1]
)
def test_whole_sequence_matches_the_streaming_replay(length):
    transformer = small_transformer()
    x = torch.randn(1, SMALL.dim, length)
    state = transformer.init_state(batch_size=1)
    with torch.no_grad():
        whole = transformer(x)
        chunked = torch.cat(
            [
                transformer.step(x[..., t : t + SMALL.frame_ratio], state)
                for t in range(0, x.shape[-1], SMALL.frame_ratio)
            ],
            -1,
        )
    torch.testing.assert_close(whole, chunked, atol=1e-6, rtol=1e-5)


def compare_staggered_decode_batches(
    codec: MimiCodec, slots: MimiDecodeSlots, *, use_batched_reference: bool = False
) -> None:
    generator = torch.Generator().manual_seed(12)
    stream_lengths = [
        codec.spec.context // codec.spec.frame_ratio + 17 + 3 * i for i in range(8)
    ]
    codes = [
        torch.randint(
            codec.spec.codebook_size,
            (1, codec.spec.num_codebooks, frames),
            generator=generator,
        ).to(codec.device)
        for frames in stream_lengths
    ]
    for _ in range(2):
        decode_slots = [slots.acquire() for _ in range(8)]
        reference_pool = (
            codec.init_decode_state(batch_size=8) if use_batched_reference else None
        )
        references = (
            []
            if use_batched_reference
            else [codec.init_decode_state(batch_size=1) for _ in range(8)]
        )
        positions = [0] * 8
        tick = 0
        while positions != stream_lengths:
            groups: dict[int, list[int]] = {}
            for stream_index in reversed(range(8)):
                if (
                    tick >= stream_index
                    and positions[stream_index] < stream_lengths[stream_index]
                ):
                    width = min(
                        tick % 3 + 1,
                        stream_lengths[stream_index] - positions[stream_index],
                    )
                    groups.setdefault(width, []).append(stream_index)
                else:
                    pass
            for width, stream_indices in groups.items():
                chunks = [
                    codes[i][..., positions[i] : positions[i] + width]
                    for i in stream_indices
                ]
                batch_codes = torch.cat(chunks)
                waveforms = slots.decode_batch(
                    batch_codes, [decode_slots[i] for i in stream_indices]
                ).clone()
                if reference_pool is not None:
                    row_indices = torch.tensor(stream_indices, device=codec.device)
                    reference_state = reference_pool.select_rows(row_indices)
                    expected = codec.decode_step(batch_codes, reference_state)
                    reference_pool.copy_rows(row_indices, reference_state)
                    torch.testing.assert_close(waveforms, expected, atol=0, rtol=0)
                else:
                    for batch_row, stream_index in enumerate(stream_indices):
                        expected = codec.decode_step(
                            chunks[batch_row], references[stream_index]
                        )
                        torch.testing.assert_close(
                            waveforms[batch_row : batch_row + 1],
                            expected,
                            atol=1e-5,
                            rtol=1e-5,
                        )
                for stream_index in stream_indices:
                    positions[stream_index] += width
            tick += 1
        for slot in decode_slots:
            slots.release(slot)


def test_platforms_without_codec_decode_graphs_decode_and_reuse_fresh_states(
    small_codec: MimiCodec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mimi_decode_graph,
        "current_platform",
        SimpleNamespace(enable_codec_decode_graph=lambda: False),
    )
    slots = MimiDecodeSlots(small_codec, num_slots=8, max_graph_frames=1)
    compare_staggered_decode_batches(small_codec, slots)


@pytest.mark.accelerator
@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.skipif(
    not current_platform.enable_codec_decode_graph(),
    reason="the platform does not capture codec decode graphs",
)
def test_graph_replay_matches_eager_decode_across_staggered_batches_and_slot_reuse(
    random_codec: MimiCodec, allow_tf32: bool
) -> None:
    # Note (edwardzh): TF32 convolution rounding depends on batch size.
    with torch.backends.cudnn.flags(allow_tf32=allow_tf32):
        codec = random_codec.cuda()
        slots = MimiDecodeSlots(codec, num_slots=8, max_graph_frames=2)
        compare_staggered_decode_batches(codec, slots, use_batched_reference=allow_tf32)
        pooled = [slots.acquire() for _ in range(8)]
        overflow = slots.acquire()
        codes = torch.zeros(2, 8, 1, dtype=torch.long, device=codec.device)
        expected = codec.decode(codes)
        torch.testing.assert_close(
            slots.decode_batch(codes, [pooled[5], overflow]),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )
        for slot in [*pooled, overflow]:
            slots.release(slot)
