# SPDX-License-Identifier: Apache-2.0
"""Mimi streaming state with selectable rows for batched decode."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.components.causal_conv import (
    ConvState,
    ConvTransposeState,
)
from sglang_omni.models.personaplex.components.mimi_transformer import (
    AttentionState,
    TransformerState,
)

SEANetLayerState = ConvState | ConvTransposeState | None
SEANetState = list[SEANetLayerState | list[SEANetLayerState]]


def stack_tensors(state: SEANetState) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for module_state in state:
        if isinstance(module_state, list):
            tensors.extend(stack_tensors(module_state))
        elif isinstance(module_state, ConvState):
            tensors.extend((module_state.previous, module_state.has_started))
        elif isinstance(module_state, ConvTransposeState):
            tensors.append(module_state.partial)
        else:
            pass
    return tensors


def select_stack_rows(
    state: SEANetState, row_indices: torch.Tensor | slice
) -> SEANetState:
    selected_state: SEANetState = []
    for module_state in state:
        if isinstance(module_state, list):
            selected_state.append(select_stack_rows(module_state, row_indices))
        elif isinstance(module_state, ConvState):
            selected_state.append(
                ConvState(
                    previous=module_state.previous[row_indices],
                    has_started=module_state.has_started[row_indices],
                )
            )
        elif isinstance(module_state, ConvTransposeState):
            selected_state.append(
                ConvTransposeState(partial=module_state.partial[row_indices])
            )
        else:
            selected_state.append(None)
    return selected_state


@dataclass(kw_only=True)
class MimiEncodeState:
    encoder: SEANetState
    transformer: TransformerState
    downsample: ConvState


@dataclass(kw_only=True)
class MimiDecodeState:
    upsample: ConvTransposeState
    transformer: TransformerState
    decoder: SEANetState

    def reset(self) -> None:
        """Reset in place so captured graphs keep their state addresses."""
        for tensor in self.tensors():
            tensor.zero_()

    def tensors(self) -> list[torch.Tensor]:
        return (
            [self.upsample.partial, self.transformer.offset]
            + [
                tensor
                for layer in self.transformer.layers
                for tensor in (layer.keys, layer.values)
            ]
            + stack_tensors(self.decoder)
        )

    def select_rows(self, row_indices: torch.Tensor | slice) -> MimiDecodeState:
        """A tensor index copies rows; a slice preserves views into the pool."""
        return MimiDecodeState(
            upsample=ConvTransposeState(partial=self.upsample.partial[row_indices]),
            transformer=TransformerState(
                offset=self.transformer.offset[row_indices],
                layers=[
                    AttentionState(
                        keys=layer.keys[row_indices],
                        values=layer.values[row_indices],
                    )
                    for layer in self.transformer.layers
                ],
            ),
            decoder=select_stack_rows(self.decoder, row_indices),
        )

    def copy_rows(self, row_indices: torch.Tensor, source: MimiDecodeState) -> None:
        for destination_tensor, source_tensor in zip(
            self.tensors(), source.tensors(), strict=True
        ):
            destination_tensor.index_copy_(0, row_indices, source_tensor)
