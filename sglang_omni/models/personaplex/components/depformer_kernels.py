# SPDX-License-Identifier: Apache-2.0
"""Triton pointwise kernels for contiguous PersonaPlex batched inference."""

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from triton.compiler import CompiledKernel

from sglang_omni.models.personaplex.architecture import DEPFORMER

NORM_VECTOR_ELEMENTS = 4
NORM_THREADS = DEPFORMER.dim // NORM_VECTOR_ELEMENTS
WARP_THREADS = 32
GATE_BLOCK_ELEMENTS = 256
GATE_BLOCK_COUNT = (
    DEPFORMER.ffn_hidden + GATE_BLOCK_ELEMENTS - 1
) // GATE_BLOCK_ELEMENTS
# note (Codex): Triton specializes tensor pointers on 16-byte alignment.
POINTER_ALIGNMENT_BYTES = 16
SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    (torch.float32, torch.bfloat16, torch.float16)
)
RMS_NORM_KERNELS: dict[
    tuple[torch.dtype, torch.dtype, int, int, bool, bool], CompiledKernel
] = {}
SILU_GATE_KERNELS: dict[tuple[torch.dtype, int, bool], CompiledKernel] = {}


@triton.jit(do_not_specialize=["epsilon"])
def rms_norm_f32_kernel(
    hidden_states: tl.tensor,
    alpha: tl.tensor,
    normalized_states: tl.tensor,
    epsilon: float,
    norm_threads: tl.constexpr,
    reduction_stages: tl.constexpr,
    dimension: tl.constexpr,
    vector_elements: tl.constexpr,
) -> None:
    row_index = tl.program_id(0)
    lane_indices = tl.arange(0, norm_threads)
    lane_sums = tl.full((norm_threads,), 0, tl.float32)
    # note (Codex): Matching eager's FP32 accumulation order preserves BF16 rounding.
    for vector_index in tl.static_range(vector_elements):
        vector_sums = tl.full((norm_threads,), 0, tl.float32)
        for chunk_index in tl.static_range(
            dimension // (norm_threads * vector_elements)
        ):
            column_indices = (
                lane_indices + chunk_index * norm_threads
            ) * vector_elements + vector_index
            hidden_values = tl.load(
                hidden_states + row_index * dimension + column_indices
            ).to(tl.float32)
            squared_values = libdevice.mul_rn(hidden_values, hidden_values)
            vector_sums = libdevice.add_rn(vector_sums, squared_values)
        lane_sums = libdevice.add_rn(lane_sums, vector_sums)
    for reduction_stage in tl.static_range(reduction_stages):
        reduction_offset = norm_threads >> (reduction_stage + 1)
        shuffled_sums = tl.gather(lane_sums, lane_indices ^ reduction_offset, axis=0)
        lane_sums = libdevice.add_rn(lane_sums, shuffled_sums)
    row_sum = tl.gather(lane_sums, tl.full((1,), 0, tl.int32), axis=0)
    variance = libdevice.add_rn(libdevice.mul_rn(row_sum, 1.0 / dimension), epsilon)
    inverse_rms = libdevice.rsqrt(variance)
    column_indices = tl.arange(0, dimension)
    hidden_values = tl.load(hidden_states + row_index * dimension + column_indices).to(
        tl.float32
    )
    alpha_values = tl.load(alpha + column_indices).to(tl.float32)
    normalized_values = libdevice.mul_rn(
        hidden_values, libdevice.mul_rn(alpha_values, inverse_rms)
    )
    tl.store(
        normalized_states + row_index * dimension + column_indices,
        normalized_values,
    )


@triton.jit
def silu_gate_kernel(
    gate_up: tl.tensor,
    activated_states: tl.tensor,
    hidden_dimension: tl.constexpr,
    block_elements: tl.constexpr,
) -> None:
    row_index = tl.program_id(0)
    column_indices = tl.program_id(1) * block_elements + tl.arange(0, block_elements)
    column_mask = column_indices < hidden_dimension
    gate = tl.load(
        gate_up + row_index * 2 * hidden_dimension + column_indices,
        mask=column_mask,
        other=0,
    ).to(tl.float32)
    up = tl.load(
        gate_up + row_index * 2 * hidden_dimension + hidden_dimension + column_indices,
        mask=column_mask,
        other=0,
    ).to(tl.float32)
    activated_gate = libdevice.div_rn(gate, libdevice.add_rn(1.0, libdevice.exp(-gate)))
    # note (Codex): Eager SiLU rounds before the separate multiplication kernel.
    rounded_gate = activated_gate.to(gate_up.dtype.element_ty).to(tl.float32)
    activated_values = libdevice.mul_rn(rounded_gate, up)
    tl.store(
        activated_states + row_index * hidden_dimension + column_indices,
        activated_values,
        mask=column_mask,
    )


def fused_rms_norm_f32(
    hidden_states: torch.Tensor, alpha: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Normalize contiguous production inputs row by row with FP32 arithmetic."""
    assert not torch.is_grad_enabled()
    assert hidden_states.is_cuda and hidden_states.dtype in SUPPORTED_DTYPES
    assert hidden_states.ndim == 2 and hidden_states.shape[1] == DEPFORMER.dim
    assert hidden_states.shape[0] > 0 and hidden_states.is_contiguous()
    assert alpha.shape == (DEPFORMER.dim,) and alpha.is_contiguous()
    assert alpha.device == hidden_states.device and alpha.dtype in SUPPORTED_DTYPES
    normalized_states = torch.empty_like(hidden_states)
    # note (Codex): PyTorch reduces each row with fewer lanes as the batch grows.
    norm_threads = max(
        WARP_THREADS, NORM_THREADS >> max(0, hidden_states.shape[0].bit_length() - 2)
    )
    device_index = hidden_states.get_device()
    kernel_key = (
        hidden_states.dtype,
        alpha.dtype,
        device_index,
        norm_threads,
        hidden_states.data_ptr() % POINTER_ALIGNMENT_BYTES == 0,
        alpha.data_ptr() % POINTER_ALIGNMENT_BYTES == 0,
    )
    normalize = RMS_NORM_KERNELS.get(kernel_key)
    grid = (hidden_states.shape[0], 1, 1)
    kernel_arguments = (
        hidden_states,
        alpha,
        normalized_states,
        epsilon,
        norm_threads,
        norm_threads.bit_length() - 1,
        DEPFORMER.dim,
        NORM_VECTOR_ELEMENTS,
    )
    with torch.cuda.device(device_index):
        if normalize is None or rms_norm_f32_kernel.pre_run_hooks:
            RMS_NORM_KERNELS[kernel_key] = rms_norm_f32_kernel[grid](
                *kernel_arguments,
                num_warps=norm_threads // WARP_THREADS,
                enable_fp_fusion=False,
                enable_reflect_ftz=False,
            )
        else:
            normalize[grid](*kernel_arguments)
    return normalized_states


def fused_silu_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """Activate contiguous production projections with eager rounding."""
    assert not torch.is_grad_enabled()
    assert gate_up.is_cuda and gate_up.dtype in SUPPORTED_DTYPES
    assert gate_up.ndim == 2 and gate_up.shape[1] == 2 * DEPFORMER.ffn_hidden
    assert gate_up.shape[0] > 0 and gate_up.is_contiguous()
    activated_states = gate_up.new_empty((gate_up.shape[0], DEPFORMER.ffn_hidden))
    device_index = gate_up.get_device()
    kernel_key = (
        gate_up.dtype,
        device_index,
        gate_up.data_ptr() % POINTER_ALIGNMENT_BYTES == 0,
    )
    activate = SILU_GATE_KERNELS.get(kernel_key)
    grid = (gate_up.shape[0], GATE_BLOCK_COUNT, 1)
    kernel_arguments = (
        gate_up,
        activated_states,
        DEPFORMER.ffn_hidden,
        GATE_BLOCK_ELEMENTS,
    )
    with torch.cuda.device(device_index):
        if activate is None or silu_gate_kernel.pre_run_hooks:
            SILU_GATE_KERNELS[kernel_key] = silu_gate_kernel[grid](
                *kernel_arguments,
                num_warps=GATE_BLOCK_ELEMENTS // WARP_THREADS,
                enable_fp_fusion=False,
                enable_reflect_ftz=False,
            )
        else:
            activate[grid](*kernel_arguments)
    return activated_states
