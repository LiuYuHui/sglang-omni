# SPDX-License-Identifier: Apache-2.0
"""CPU tests for component attribution and overlapping Chrome trace activity."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from benchmarks.eval.personaplex_profiling_report import (
    ChromeTrace,
    TraceArguments,
    TraceEvent,
    build_component_report,
)


def trace_event(
    category: str,
    start_microseconds: float,
    duration_microseconds: float,
    *,
    name: str = "operation",
    process_id: int = 101,
    thread_id: int = 7,
    external_id: int | None = None,
    correlation: int | None = None,
    device_id: int | None = None,
) -> TraceEvent:
    return TraceEvent(
        ph="X",
        cat=category,
        name=name,
        pid=process_id,
        tid=thread_id,
        ts=start_microseconds,
        dur=duration_microseconds,
        args=TraceArguments(
            **{
                "External id": external_id,
                "correlation": correlation,
                "device": device_id,
            }
        ),
    )


def write_trace(
    trace_path: Path,
    events: list[TraceEvent],
    *,
    base_time_nanoseconds: int = 0,
) -> Path:
    trace = ChromeTrace(traceEvents=events, baseTimeNanoseconds=base_time_nanoseconds)
    serialized = trace.model_dump_json(by_alias=True, exclude_none=True)
    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "wt", encoding="utf-8") as trace_file:
            trace_file.write(serialized)
    else:
        trace_path.write_text(serialized)
    return trace_path


def test_external_id_links_runtime_worker_thread_and_unions_overlap(
    tmp_path: Path,
) -> None:
    trace_path = write_trace(
        tmp_path / "trace.json",
        [
            trace_event(
                "user_annotation", 0, 5000, name="personaplex.temporal_transformer"
            ),
            trace_event("user_annotation", 1000, 2000, name="personaplex.depformer"),
            trace_event("cpu_op", 100, 800, external_id=10),
            trace_event("cpu_op", 1500, 400, external_id=20),
            trace_event(
                "cuda_runtime",
                300,
                100,
                name="cudaLaunchKernel",
                thread_id=999,
                external_id=10,
                correlation=1,
            ),
            trace_event(
                "cuda_runtime",
                1600,
                200,
                name="cudaLaunchKernel",
                thread_id=999,
                external_id=20,
                correlation=2,
            ),
            trace_event(
                "cuda_runtime",
                600,
                100,
                name="cudaLaunchKernel",
                thread_id=999,
                external_id=10,
                correlation=3,
            ),
            trace_event(
                "kernel",
                400,
                2000,
                process_id=0,
                external_id=10,
                correlation=1,
                device_id=0,
            ),
            trace_event(
                "kernel",
                1800,
                2000,
                process_id=0,
                external_id=20,
                correlation=2,
                device_id=0,
            ),
            trace_event(
                "kernel",
                2000,
                600,
                process_id=0,
                external_id=10,
                correlation=3,
                device_id=0,
            ),
            trace_event(
                "gpu_user_annotation",
                0,
                100000,
                name="personaplex.temporal_transformer",
                process_id=0,
            ),
            trace_event(
                "user_annotation",
                0,
                100000,
                name="personaplex.depformer",
                process_id=0,
                device_id=0,
            ),
            trace_event(
                "cuda_runtime", 6000, 10000, name="cudaDeviceSynchronize", thread_id=999
            ),
        ],
    )
    report = build_component_report([trace_path])
    components = {component["name"]: component for component in report["components"]}
    temporal = components["personaplex.temporal_transformer"]
    depformer = components["personaplex.depformer"]
    assert temporal["cpu_scope_milliseconds"] == 5.0
    assert temporal["scope_count"] == 1
    assert temporal["gpu_kernel_milliseconds"] == 2.2
    assert temporal["cuda_launch_milliseconds"] == 0.2
    assert temporal["cuda_synchronization_milliseconds"] == 0.0
    assert depformer["cpu_scope_milliseconds"] == 2.0
    assert depformer["gpu_kernel_milliseconds"] == 2.0
    assert report["cpu_scope_busy_milliseconds"] == 5.0
    assert report["gpu_kernel_milliseconds"] == 3.4
    assert report["kernel_attribution_fraction"] == 1.0
    assert report["devices"][0]["gpu_busy_milliseconds"] == 3.4
    assert report["devices"][0]["gpu_idle_milliseconds"] == pytest.approx(1.6)
    assert report["dominant_gpu_components"][:2] == [
        "personaplex.temporal_transformer",
        "personaplex.depformer",
    ]
    assert json.loads(json.dumps(report)) == report


def test_correlation_fallback_transfers_and_scoped_synchronization(
    tmp_path: Path,
) -> None:
    trace_path = write_trace(
        tmp_path / "trace.json.gz",
        [
            trace_event("user_annotation", 0, 2000, name="personaplex.h2d"),
            trace_event("cpu_op", 100, 1500, external_id=22),
            trace_event(
                "cuda_runtime",
                200,
                300,
                name="cudaMemcpyAsync",
                thread_id=555,
                external_id=22,
                correlation=5,
            ),
            trace_event(
                "cuda_runtime",
                500,
                900,
                name="cudaStreamSynchronize",
                thread_id=555,
                external_id=22,
            ),
            trace_event(
                "gpu_memcpy", 400, 700, process_id=0, correlation=5, device_id=0
            ),
            trace_event("kernel", 500, 300, process_id=0, correlation=9, device_id=0),
            trace_event("gpu_memcpy", 1500, 200, process_id=0, device_id=0),
        ],
    )
    report = build_component_report([trace_path])
    transfer = next(
        component
        for component in report["components"]
        if component["name"] == "personaplex.h2d"
    )
    assert transfer["gpu_transfer_milliseconds"] == 0.7
    assert transfer["cuda_launch_milliseconds"] == 0.0
    assert transfer["cuda_synchronization_milliseconds"] == 0.9
    assert report["gpu_transfer_milliseconds"] == 0.9
    assert report["unattributed_gpu_transfer_milliseconds"] == 0.2
    assert report["unattributed_gpu_kernel_milliseconds"] == 0.3
    assert report["transfer_attribution_fraction"] == 0.5
    assert report["kernel_attribution_fraction"] == 0.0
    assert report["devices"][0]["gpu_busy_milliseconds"] == 0.9
    assert report["devices"][0]["gpu_idle_milliseconds"] == 1.1


def test_trace_and_process_namespaces_prevent_external_id_collisions(
    tmp_path: Path,
) -> None:
    first_trace = write_trace(
        tmp_path / "first.json",
        [
            trace_event("user_annotation", 0, 1000, name="personaplex.mimi_encode"),
            trace_event("cpu_op", 100, 300, external_id=1),
            trace_event(
                "cuda_runtime",
                200,
                100,
                name="cudaLaunchKernel",
                external_id=1,
                correlation=1,
            ),
            trace_event(
                "kernel",
                300,
                400,
                process_id=0,
                external_id=1,
                correlation=1,
                device_id=0,
            ),
        ],
    )
    second_trace = write_trace(
        tmp_path / "second.json",
        [
            trace_event("user_annotation", 0, 1000, name="personaplex.mimi_decode"),
            trace_event("cpu_op", 100, 300, external_id=1),
            trace_event(
                "cuda_runtime",
                200,
                100,
                name="cudaLaunchKernel",
                external_id=1,
                correlation=1,
            ),
            trace_event(
                "kernel",
                300,
                600,
                process_id=0,
                external_id=1,
                correlation=1,
                device_id=0,
            ),
            trace_event(
                "user_annotation", 0, 1000, name="personaplex.depformer", process_id=202
            ),
            trace_event("cpu_op", 100, 300, process_id=202, external_id=1),
            trace_event(
                "cuda_runtime",
                200,
                100,
                name="cudaLaunchKernel",
                process_id=202,
                external_id=1,
                correlation=2,
            ),
            trace_event(
                "kernel",
                900,
                100,
                process_id=0,
                external_id=1,
                correlation=2,
                device_id=0,
            ),
        ],
    )
    report = build_component_report([first_trace, second_trace])
    components = {component["name"]: component for component in report["components"]}
    assert components["personaplex.mimi_encode"]["gpu_kernel_milliseconds"] == 0.4
    assert components["personaplex.mimi_decode"]["gpu_kernel_milliseconds"] == 0.6
    assert components["personaplex.depformer"]["gpu_kernel_milliseconds"] == 0.1
    assert report["gpu_kernel_milliseconds"] == 0.7
    assert report["devices"][0]["gpu_busy_milliseconds"] == 0.7
    assert report["kernel_attribution_fraction"] == 1.0


def test_missing_scopes_preserve_unattributed_costs(tmp_path: Path) -> None:
    trace_path = write_trace(
        tmp_path / "unannotated.json",
        [
            trace_event("kernel", 100, 300, process_id=0, device_id=0),
            trace_event("kernel", 200, 500, process_id=0, device_id=0),
        ],
    )
    report = build_component_report([trace_path])
    assert len(report["missing_components"]) == 8
    assert all(
        component["cpu_scope_milliseconds"] is None
        for component in report["components"]
    )
    assert all(
        component["gpu_kernel_milliseconds"] is None
        for component in report["components"]
    )
    assert report["gpu_kernel_milliseconds"] == 0.6
    assert report["unattributed_gpu_kernel_milliseconds"] == 0.6
    assert report["kernel_attribution_fraction"] == 0.0
    assert report["transfer_attribution_fraction"] is None
    assert report["devices"][0]["window_source"] == "gpu_events"
    assert report["devices"][0]["gpu_idle_milliseconds"] == 0.0


def test_ambiguous_process_correlation_is_left_unattributed(tmp_path: Path) -> None:
    events: list[TraceEvent] = []
    for process_id, component_name in (
        (101, "personaplex.mimi_encode"),
        (202, "personaplex.mimi_decode"),
    ):
        events.extend(
            [
                trace_event(
                    "user_annotation",
                    0,
                    1000,
                    name=component_name,
                    process_id=process_id,
                ),
                trace_event("cpu_op", 100, 300, process_id=process_id, external_id=1),
                trace_event(
                    "cuda_runtime",
                    200,
                    100,
                    name="cudaLaunchKernel",
                    process_id=process_id,
                    external_id=1,
                    correlation=1,
                ),
            ]
        )
    events.append(
        trace_event(
            "kernel", 300, 400, process_id=0, external_id=1, correlation=1, device_id=0
        )
    )
    report = build_component_report([write_trace(tmp_path / "ambiguous.json", events)])
    assert report["kernel_attribution_fraction"] == 0.0
    assert report["unattributed_gpu_kernel_milliseconds"] == 0.4
    assert all(component["kernel_count"] == 0 for component in report["components"])


def test_memset_is_separate_from_transfers_and_included_in_gpu_busy(
    tmp_path: Path,
) -> None:
    trace_path = write_trace(
        tmp_path / "memset.json",
        [
            trace_event("user_annotation", 0, 2000, name="personaplex.mimi_decode"),
            trace_event("cpu_op", 100, 1300, external_id=1),
            trace_event(
                "cuda_runtime",
                200,
                100,
                name="cudaMemsetAsync",
                external_id=1,
                correlation=2,
            ),
            trace_event(
                "gpu_memset",
                300,
                500,
                process_id=0,
                external_id=1,
                correlation=2,
                device_id=0,
            ),
            trace_event("kernel", 500, 700, process_id=0, external_id=1, device_id=0),
            trace_event("gpu_memcpy", 1500, 200, process_id=0, device_id=0),
            trace_event("gpu_memset", 1000, 100, process_id=0, device_id=0),
        ],
    )
    report = build_component_report([trace_path])
    decoder = next(
        component
        for component in report["components"]
        if component["name"] == "personaplex.mimi_decode"
    )
    assert decoder["gpu_transfer_milliseconds"] == 0.0
    assert decoder["transfer_count"] == 0
    assert decoder["gpu_memset_milliseconds"] == 0.5
    assert report["gpu_transfer_milliseconds"] == 0.2
    assert report["gpu_memset_milliseconds"] == 0.6
    assert report["unattributed_gpu_memset_milliseconds"] == 0.1
    assert report["transfer_attribution_fraction"] == 0.0
    assert report["devices"][0]["gpu_transfer_milliseconds"] == 0.2
    assert report["devices"][0]["gpu_memset_milliseconds"] == 0.6
    assert report["devices"][0]["gpu_busy_milliseconds"] == 1.1
    assert report["devices"][0]["gpu_idle_milliseconds"] == pytest.approx(0.9)


def test_trace_base_clocks_are_aligned_before_device_union(tmp_path: Path) -> None:
    first_trace = write_trace(
        tmp_path / "first.json",
        [
            trace_event("kernel", 1000, 1000, process_id=0, device_id=0),
        ],
        base_time_nanoseconds=1_000_000,
    )
    second_trace = write_trace(
        tmp_path / "second.json",
        [
            trace_event("kernel", 0, 1000, process_id=0, device_id=0),
        ],
        base_time_nanoseconds=2_500_000,
    )
    report = build_component_report([first_trace, second_trace])
    assert report["gpu_kernel_milliseconds"] == 1.5
    assert report["devices"][0]["gpu_busy_milliseconds"] == 1.5


def test_empty_trace_inputs_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="At least one Chrome trace"):
        build_component_report([])
