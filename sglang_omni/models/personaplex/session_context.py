# SPDX-License-Identifier: Apache-2.0
"""Grow position metadata while retaining the streaming session's window KV pool."""

from __future__ import annotations

import gc
import logging

import torch
from sglang.srt.layers.rotary_embedding import RotaryEmbedding

from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner

logger = logging.getLogger(__name__)


class PersonaPlexSessionContext:
    def __init__(self, model_runner: SGLModelRunner) -> None:
        self.model_runner: SGLModelRunner = model_runner

    def ensure_capacity(self, required_positions: int) -> None:
        """Grow absolute-position metadata after all launched work has drained."""
        model_runner = self.model_runner
        request_pool = model_runner.req_to_token_pool
        position_capacity = model_runner.model_config.context_len
        if required_positions <= position_capacity:
            return
        else:
            pass

        new_position_capacity = max(required_positions, position_capacity * 2)
        mapping_padding = request_pool.max_context_len - position_capacity
        new_mapping_capacity = new_position_capacity + mapping_padding
        old_token_mapping = request_pool.req_to_token
        torch.cuda.synchronize(old_token_mapping.device)
        if gc.get_freeze_count():
            # note (Codex): Startup freezing pins obsolete backend cycles; refreezing would pin live sessions instead.
            gc.unfreeze()
        else:
            pass

        # note (Codex): Captured backbone graphs retain the old mapping and rotary-cache addresses.
        model_runner.decode_cuda_graph_runner = None
        model_runner.prefill_cuda_graph_runner = None
        model_runner.decode_cuda_graph_captured = False
        model_runner.decode_cuda_graph_capture_bs = []

        new_token_mapping = torch.zeros(
            (old_token_mapping.shape[0], new_mapping_capacity),
            dtype=old_token_mapping.dtype,
            device=old_token_mapping.device,
        )
        new_token_mapping[:, : old_token_mapping.shape[1]].copy_(old_token_mapping)
        for module in model_runner.model.modules():
            if isinstance(module, RotaryEmbedding):
                # note (Codex): Reserved rows can use CPU-rounded frequencies; regenerate unused positions on the cache device.
                module.cos_sin_cache = module.cos_sin_cache[:position_capacity]
                with torch.device(module.cos_sin_cache.device):
                    module._ensure_cos_sin_cache_length(  # noqa: leading-underscore - RotaryEmbedding has no public growth API.
                        new_position_capacity - 1
                    )
                module.max_position_embeddings = new_position_capacity
            else:
                pass

        request_pool.req_to_token = new_token_mapping
        request_pool.max_context_len = new_mapping_capacity
        model_runner.kv_index_translator.req_to_token = new_token_mapping
        model_runner.model_config.context_len = new_position_capacity
        model_runner.model_config.hf_config.max_position_embeddings = (
            new_position_capacity
        )
        model_runner.model_config.hf_text_config.max_position_embeddings = (
            new_position_capacity
        )
        model_runner.init_attention_backends()
        gc.collect()
        logger.info(
            f"PersonaPlex session position capacity grew from {position_capacity} "
            f"to {new_position_capacity}; backbone execution uses eager mode"
        )
