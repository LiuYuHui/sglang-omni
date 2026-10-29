# SPDX-License-Identifier: Apache-2.0
"""Checkpoint mapping and growing session metadata preserve model computations."""

import subprocess
import sys
from dataclasses import replace
from textwrap import dedent
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.rotary_embedding import base as rotary_embedding_base
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from torch import nn
from transformers import PretrainedConfig

from sglang_omni.models.personaplex.architecture import (
    AUDIO_CARD,
    NUM_AUDIO_STREAMS,
    NUM_STREAMS,
    TEMPORAL_TRANSFORMER,
    TEXT_CARD,
)
from sglang_omni.models.personaplex.session_context import PersonaPlexSessionContext
from sglang_omni.models.personaplex.sglang_model import (
    PersonaPlexForCausalLM,
    backbone_weight,
)

DIM = 4


def test_backbone_tensors_map_onto_llama_names():
    in_proj = torch.arange(12.0).view(6, 2)
    mapped = dict(
        backbone_weight("transformer.layers.3.self_attn.in_proj_weight", in_proj)
    )
    q, k, v = in_proj.chunk(3)
    assert torch.equal(mapped["model.layers.3.self_attn.q_proj.weight"], q)
    assert torch.equal(mapped["model.layers.3.self_attn.k_proj.weight"], k)
    assert torch.equal(mapped["model.layers.3.self_attn.v_proj.weight"], v)

    linear_in = torch.arange(8.0).view(4, 2)
    mapped = dict(
        backbone_weight("transformer.layers.0.gating.linear_in.weight", linear_in)
    )
    assert torch.equal(mapped["model.layers.0.mlp.gate_proj.weight"], linear_in[:2])
    assert torch.equal(mapped["model.layers.0.mlp.up_proj.weight"], linear_in[2:])

    alpha = torch.ones(1, 1, DIM)
    ((name, tensor),) = backbone_weight("transformer.layers.1.norm2.alpha", alpha)
    assert name == "model.layers.1.post_attention_layernorm.weight"
    assert tensor.shape == (DIM,)
    ((name, _),) = backbone_weight(
        "transformer.layers.1.self_attn.out_proj.weight", torch.zeros(DIM, DIM)
    )
    assert name == "model.layers.1.self_attn.o_proj.weight"
    with pytest.raises(KeyError):
        list(backbone_weight("transformer.layers.1.unknown.weight", alpha))


def fake_model() -> SimpleNamespace:
    loaded = SimpleNamespace(backbone=None, depformer=None)
    return SimpleNamespace(
        loaded=loaded,
        temporal=replace(TEMPORAL_TRANSFORMER, num_layers=1),
        llm=SimpleNamespace(load_weights=lambda w: setattr(loaded, "backbone", w)),
        audio_emb=nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, DIM) for _ in range(NUM_AUDIO_STREAMS)
        ),
        text_emb=nn.Embedding(TEXT_CARD + 1, DIM),
        depformer=SimpleNamespace(
            load_reference_weights=lambda w: setattr(loaded, "depformer", w)
        ),
    )


def test_load_weights_routes_every_checkpoint_group():
    model = fake_model()
    text_emb = torch.randn(TEXT_CARD + 1, DIM)
    audio = {f"emb.{k}.weight": torch.randn(AUDIO_CARD + 1, DIM) for k in range(16)}
    weights = {
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(DIM, DIM),
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * DIM, DIM),
        "transformer.layers.0.norm1.alpha": torch.ones(1, 1, DIM),
        "transformer.layers.0.norm2.alpha": torch.ones(1, 1, DIM),
        "transformer.layers.0.gating.linear_in.weight": torch.zeros(2 * DIM, DIM),
        "transformer.layers.0.gating.linear_out.weight": torch.zeros(DIM, DIM),
        "out_norm.alpha": torch.ones(1, 1, DIM),
        "text_linear.weight": torch.randn(TEXT_CARD, DIM),
        "text_emb.weight": text_emb,
        "depformer_in.0.weight": torch.zeros(1),
        "linears.0.weight": torch.zeros(1),
        **audio,
    }

    for missing in (
        "transformer.layers.0.self_attn.in_proj_weight",
        "transformer.layers.0.gating.linear_out.weight",
        "out_norm.alpha",
        "text_linear.weight",
        "text_emb.weight",
    ):
        incomplete = {
            name: tensor for name, tensor in weights.items() if name != missing
        }
        rejected = fake_model()
        with pytest.raises(ValueError, match="missing PersonaPlex backbone weights"):
            PersonaPlexForCausalLM.load_weights(rejected, incomplete.items())
        assert rejected.loaded.backbone is None

    PersonaPlexForCausalLM.load_weights(model, weights.items())

    backbone = dict(model.loaded.backbone)
    assert set(backbone) == {
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    }
    assert torch.equal(backbone["model.embed_tokens.weight"], text_emb[:TEXT_CARD])
    assert torch.equal(model.text_emb.weight, text_emb)
    assert torch.equal(model.audio_emb[5].weight, audio["emb.5.weight"])
    assert set(model.loaded.depformer) == {"depformer_in.0.weight", "linears.0.weight"}

    with pytest.raises(KeyError, match="unexpected PersonaPlex tensor"):
        PersonaPlexForCausalLM.load_weights(fake_model(), [("mystery", torch.zeros(1))])


def test_embed_rows_reads_text_from_column_zero_and_codebook_k_from_column_k_plus_one():
    model = SimpleNamespace(
        audio_emb=nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, NUM_STREAMS) for _ in range(NUM_AUDIO_STREAMS)
        ),
        text_emb=nn.Embedding(TEXT_CARD + 1, NUM_STREAMS),
    )
    with torch.no_grad():
        for k, table in enumerate(model.audio_emb):
            table.weight.zero_()
            table.weight[:, 1 + k] = torch.arange(AUDIO_CARD + 1, dtype=torch.float32)
        model.text_emb.weight.zero_()
        model.text_emb.weight[:, 0] = torch.arange(TEXT_CARD + 1, dtype=torch.float32)

    rows = torch.randint(0, AUDIO_CARD + 1, (3, NUM_STREAMS))
    rows[:, 0] = torch.tensor([5, 31999, 32000])
    with torch.no_grad():
        embedded = PersonaPlexForCausalLM.embed_rows(model, rows)
    assert torch.equal(embedded, rows.float())


@pytest.mark.parametrize("query_count", [1, 3], ids=["decode", "prefill"])
@pytest.mark.parametrize("boundary_offset", [-2, -1, 0, 1, 3001])
def test_attention_matches_streaming_ring_at_wraparound(
    query_count: int, boundary_offset: int
) -> None:
    capacity = TEMPORAL_TRANSFORMER.context
    last_position = capacity + boundary_offset
    positions = torch.arange(last_position + 1)
    queries = positions[-query_count:]
    model = SimpleNamespace(temporal=TEMPORAL_TRANSFORMER)
    window = PersonaPlexForCausalLM.get_attention_sliding_window_size(model)
    mask = (queries[:, None] >= positions) & (positions >= queries[:, None] - window)
    keys = torch.zeros(1, 1, len(positions), 4)
    values = positions.float().view(1, 1, -1, 1) / capacity
    query = torch.zeros(1, 1, query_count, 4)
    actual = torch.nn.functional.scaled_dot_product_attention(
        query, keys, values, attn_mask=mask
    )
    slots = torch.arange(capacity)
    for index, position in enumerate(queries.tolist()):
        end_offset = position + 1
        # note (LinzeShi): The ring labels its next overwrite slot as a future key.
        delta = slots - end_offset % capacity
        ring_positions = torch.where(
            delta <= 0, end_offset + delta, end_offset + delta - capacity
        )
        visible = (slots < end_offset) & (ring_positions <= position)
        stored_positions = position - (position - slots) % capacity
        reference_positions = stored_positions[visible].sort().values
        assert torch.equal(positions[mask[index]], reference_positions)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[:, :, index : index + 1],
            keys[:, :, reference_positions],
            values[:, :, reference_positions],
        )
        torch.testing.assert_close(actual[:, :, index : index + 1], reference)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ],
        ),
    ],
)
def test_session_growth_preserves_mapping_and_absolute_rotary_positions(
    monkeypatch: pytest.MonkeyPatch,
    device: Literal["cpu", "cuda"],
) -> None:
    """A live call must cross the initial capacity without changing retained positions."""
    if device == "cpu":
        monkeypatch.setattr(rotary_embedding_base, "_is_cpu", True)
        synchronize = Mock()
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    else:
        pass
    position_capacity, mapping_padding = 8192, 4
    head_size = TEMPORAL_TRANSFORMER.head_dim
    request_pool = ReqToTokenPool(
        2, position_capacity + mapping_padding, device, enable_memory_saver=False
    )
    request = Req(
        rid="long-call",
        origin_input_text="",
        origin_input_ids=[11, 12, 13],
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    assert request_pool.alloc([request]) is not None
    request_pool.req_to_token.copy_(
        torch.arange(request_pool.req_to_token.numel(), device=device).reshape_as(
            request_pool.req_to_token
        )
    )
    original_mapping = request_pool.req_to_token.clone()
    original_free_slots = list(request_pool.free_slots)
    request_row = request.kv.req_pool_idx
    with torch.device(device):
        rotary = RotaryEmbedding(
            head_size, head_size, position_capacity, 10000, False, torch.float32
        )
    rotary._ensure_cos_sin_cache_length(  # noqa: leading-underscore - Reproduce the upstream reserve outside the model device context.
        position_capacity + 255
    )
    original_rotary_cache = rotary.cos_sin_cache[:position_capacity].clone()
    translator = SimpleNamespace(req_to_token=request_pool.req_to_token)
    decode_graph, prefill_graph, depformer_graph = Mock(), Mock(), Mock()
    model_runner = SimpleNamespace(
        req_to_token_pool=request_pool,
        kv_index_translator=translator,
        model=torch.nn.Sequential(rotary),
        model_config=SimpleNamespace(
            context_len=position_capacity,
            hf_config=PretrainedConfig(max_position_embeddings=position_capacity),
            hf_text_config=PretrainedConfig(max_position_embeddings=position_capacity),
        ),
        decode_cuda_graph_runner=decode_graph,
        prefill_cuda_graph_runner=prefill_graph,
        depformer_cuda_graph=depformer_graph,
        decode_cuda_graph_captured=True,
        decode_cuda_graph_capture_bs=[1],
        init_attention_backends=Mock(),
    )
    session_context = PersonaPlexSessionContext(model_runner)
    session_context.ensure_capacity(position_capacity)
    if device == "cpu":
        synchronize.assert_not_called()
    else:
        pass
    model_runner.init_attention_backends.assert_not_called()
    assert model_runner.decode_cuda_graph_runner is decode_graph

    for required_positions in (position_capacity + 1, position_capacity * 2 + 1):
        session_context.ensure_capacity(required_positions)
        grown_capacity = model_runner.model_config.context_len
        assert grown_capacity >= required_positions
        assert request_pool.max_context_len == grown_capacity + mapping_padding
        assert translator.req_to_token is request_pool.req_to_token
        assert request_pool.free_slots == original_free_slots
        assert request.kv.req_pool_idx == request_row
        assert torch.equal(
            request_pool.req_to_token[:, : original_mapping.shape[1]],
            original_mapping,
        )
        assert torch.equal(
            rotary.cos_sin_cache[:position_capacity], original_rotary_cache
        )
        assert model_runner.model_config.hf_config.max_position_embeddings >= (
            required_positions
        )
        assert model_runner.model_config.hf_text_config.max_position_embeddings >= (
            required_positions
        )
        with torch.device(device):
            reference_rotary = RotaryEmbedding(
                head_size, head_size, grown_capacity, 10000, False, torch.float32
            )
        torch.testing.assert_close(
            rotary.cos_sin_cache[:grown_capacity],
            reference_rotary.cos_sin_cache,
            rtol=0,
            atol=0,
        )
        positions = torch.tensor(
            [position_capacity - 1, required_positions - 1], device=device
        )
        queries = torch.arange(
            2 * head_size, dtype=torch.float32, device=device
        ).reshape(2, head_size)
        keys = queries.flip(-1)
        rotated_queries, rotated_keys = rotary.forward_native(positions, queries, keys)
        expected_queries, expected_keys = reference_rotary.forward_native(
            positions, queries, keys
        )
        torch.testing.assert_close(rotated_queries, expected_queries, rtol=0, atol=0)
        torch.testing.assert_close(rotated_keys, expected_keys, rtol=0, atol=0)

    assert model_runner.init_attention_backends.call_count == 2
    assert model_runner.decode_cuda_graph_runner is None
    assert model_runner.prefill_cuda_graph_runner is None
    assert model_runner.decode_cuda_graph_captured is False
    assert model_runner.decode_cuda_graph_capture_bs == []
    assert model_runner.depformer_cuda_graph is depformer_graph
    request_pool.req_to_token[request_row, required_positions - 1] = 123
    assert translator.req_to_token[request_row, required_positions - 1].item() == 123
    request_pool.free(request)
    assert request_pool.available_size() == 2


def test_session_growth_releases_frozen_backends_without_freezing_calls() -> None:
    """Startup-frozen graph/backend cycles must release after a resize and call close."""
    program = dedent(
        """
        import gc
        import weakref
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        import torch
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        from sglang_omni.models.personaplex.session_context import PersonaPlexSessionContext

        resources = [Mock(), Mock(), Mock()]
        for resource in resources:
            resource.owner = resource
        resource = None
        resource_references = [weakref.ref(resource) for resource in resources]
        request_pool = ReqToTokenPool(1, 8, "cpu", enable_memory_saver=False)
        model_runner = SimpleNamespace(
            req_to_token_pool=request_pool,
            kv_index_translator=SimpleNamespace(req_to_token=request_pool.req_to_token),
            model=torch.nn.Module(),
            model_config=SimpleNamespace(
                context_len=8,
                hf_config=SimpleNamespace(max_position_embeddings=8),
                hf_text_config=SimpleNamespace(max_position_embeddings=8),
            ),
            decode_cuda_graph_runner=resources[0],
            prefill_cuda_graph_runner=resources[1],
            attn_backend=resources[2],
        )
        resources = None

        def rebuild_backend() -> None:
            model_runner.attn_backend = Mock()

        model_runner.init_attention_backends = rebuild_backend
        session_context = PersonaPlexSessionContext(model_runner)
        gc.collect()
        gc.freeze()
        assert gc.get_freeze_count() > 0
        live_session = Mock()
        live_session.owner = live_session
        session_reference = weakref.ref(live_session)
        with patch.object(torch.cuda, "synchronize"):
            session_context.ensure_capacity(9)
            assert all(reference() is None for reference in resource_references)
            assert session_reference() is live_session
            session_context.ensure_capacity(17)
        live_session = None
        gc.collect()
        assert session_reference() is None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
