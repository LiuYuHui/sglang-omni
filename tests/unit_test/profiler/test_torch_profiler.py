# SPDX-License-Identifier: Apache-2.0
"""Torch trace lifecycle and worker-thread component coverage."""

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.personaplex.profiling import component_scope
from sglang_omni.profiler import torch_profiler
from sglang_omni.profiler.torch_profiler import TorchProfiler


@pytest.fixture(autouse=True)
def reset_torch_profiler(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(torch_profiler.subprocess, "Popen", Mock())
    monkeypatch.setattr(
        torch_profiler, "profiler_activities", lambda: [ProfilerActivity.CPU]
    )
    monkeypatch.delenv("SGLANG_TORCH_PROFILER_PROFILE_ALL_THREADS", raising=False)
    TorchProfiler.stop()
    yield
    TorchProfiler.stop()


@pytest.mark.parametrize("worker_before_profile", [False, True])
def test_all_threads_records_existing_and_new_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_before_profile: bool,
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_PROFILE_ALL_THREADS", "1")
    worker_ready = threading.Event()
    start_compute = threading.Event()
    tensor_results: list[torch.Tensor] = []

    def compute() -> None:
        worker_ready.set()
        assert start_compute.wait(timeout=10)
        with component_scope("depformer"):
            tensor_results.append(torch.ones(16).sum())

    worker_thread = threading.Thread(target=compute)
    if worker_before_profile:
        worker_thread.start()
        assert worker_ready.wait(timeout=10)
    else:
        pass
    trace_path = tmp_path / "worker"
    TorchProfiler.start(str(trace_path), run_id="worker-run")
    try:
        if not worker_before_profile:
            worker_thread.start()
            assert worker_ready.wait(timeout=10)
        else:
            pass
        start_compute.set()
        worker_thread.join(timeout=10)
        assert not worker_thread.is_alive()
    finally:
        start_compute.set()
        worker_thread.join(timeout=10)
        result = TorchProfiler.stop(run_id="worker-run")
    assert result is not None
    compressed_trace_path = result["trace"]
    assert isinstance(compressed_trace_path, str)
    trace = json.loads(Path(compressed_trace_path.removesuffix(".gz")).read_text())
    assert any(
        event.get("cat") == "user_annotation"
        and event.get("name") == "personaplex.depformer"
        for event in trace["traceEvents"]
    )
    assert tensor_results[0].item() == 16


def test_component_scope_is_disabled_without_omni_profiler() -> None:
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        with component_scope("mimi_encode"):
            result = torch.ones(16).sum()
    assert result.item() == 16
    assert not any(
        event.name == "personaplex.mimi_encode" for event in profiler.events()
    )


def test_stop_exports_and_compresses_trace_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compression = Mock()
    monkeypatch.setattr(torch_profiler.subprocess, "Popen", compression)
    expected_path = TorchProfiler.start(str(tmp_path / "trace"), run_id="one-export")
    assert TorchProfiler.profiler is not None
    trace_export = Mock(wraps=TorchProfiler.profiler.export_chrome_trace)
    monkeypatch.setattr(TorchProfiler.profiler, "export_chrome_trace", trace_export)
    with component_scope("mimi_decode"):
        torch.ones(16).sum()
    result = TorchProfiler.stop(run_id="one-export")
    assert result == {"trace": expected_path, "table": None}
    json_path = expected_path.removesuffix(".gz")
    trace_export.assert_called_once_with(json_path)
    compression.assert_called_once_with(["gzip", "-f", json_path])
    assert not TorchProfiler.is_active()
    assert TorchProfiler.stop() is None


def test_replacing_profile_exports_each_session_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compression = Mock()
    monkeypatch.setattr(torch_profiler.subprocess, "Popen", compression)
    first_path = TorchProfiler.start(str(tmp_path / "first"), run_id="first")
    with component_scope("embeddings"):
        torch.ones(16).sum()
    second_path = TorchProfiler.start(str(tmp_path / "second"), run_id="second")
    with component_scope("text_logits"):
        torch.ones(16).sum()
    TorchProfiler.stop(run_id="second")
    assert compression.call_count == 2
    for trace_path in (first_path, second_path):
        assert Path(trace_path.removesuffix(".gz")).is_file()
    assert not TorchProfiler.is_active()


def test_all_threads_unsupported_build_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_PROFILE_ALL_THREADS", "1")
    monkeypatch.setattr(
        torch_profiler,
        "_ExperimentalConfig",
        Mock(side_effect=TypeError("unsupported")),
    )
    with pytest.raises(RuntimeError, match="does not support profiling all worker"):
        TorchProfiler.start(str(tmp_path / "unsupported"), run_id="unsupported")
    assert not TorchProfiler.is_active()
