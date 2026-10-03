# PersonaPlex reference evaluation

These optional scripts require CUDA and real checkpoints. They are evaluation
entry points, not pytest tests. Run them from the repository root. Install the
reference checkout into its own Python environment because its torch constraint
differs from the serving runtime.

## Greedy comparison and reproducibility

```bash
python -m benchmarks.eval.personaplex_parity \
    --reference-source /path/to/personaplex \
    --reference-python /path/to/reference-env/bin/python \
    --checkpoint /path/to/personaplex-checkpoint \
    --output-dir /path/to/reference-results
```

`--checkpoint` accepts a local directory or a model identifier supported by the
shared checkpoint resolver. Both implementations receive the same resolved weights,
tokenizer, packaged voice and caller recording. `--stage-args` forwards pipeline
overrides, such as `'--lm.engine.mem_fraction_static 0.5'`. `--atol` controls the
per-sample tolerance (default `1e-4`). `--reference-repo` only supplies the reference
CLI's config lookup; actual weights are passed as local paths.

The assistant and service cases require the complete input sample count, then
check at least 100 matching leading frames and text agreement over that prefix.
**Passing these checks does not establish full-output parity or answer quality.**
The script also checks greedy repetition and same-seed/different-seed behavior.

Every run regenerates the reference output and writes its log to the result
directory. There is no reference-output cache or mandatory revision/hash check.
Record the source and checkpoint versions with published measurements.

## Component comparison

```bash
python -m benchmarks.eval.personaplex_components \
    --reference-source /path/to/personaplex \
    --reference-python /path/to/reference-env/bin/python \
    --checkpoint /path/to/moshiko-pytorch-bf16-checkpoint \
    --dump /path/to/results/moshi-reference.safetensors
```

This uses the public `kyutai/moshiko-pytorch-bf16` base checkpoint, not the
PersonaPlex fine-tune. It compares Mimi encoding/decoding, input embeddings and
teacher-forced depformer logits in FP32/BF16. TF32 and cuDNN autotuning are disabled.
The final depformer step deliberately diagnoses the reference ring difference;
its emulated result is not proof that the unmodified implementations match.

`personaplex_reference_dump.py` runs under the reference interpreter and regenerates
the dump on every invocation. Large dumps, checkpoints and generated audio belong
in external result directories.

## Component cost profiling

```bash
python -m benchmarks.eval.personaplex_profiling \
    --model-path /path/to/personaplex-checkpoint \
    --audio /path/to/caller.wav \
    --output-dir /path/to/profiling-results \
    --requests 2 --concurrency 2 \
    --lm.engine.mem_fraction_static 0.8 \
    --lm.engine.context_length 2048 \
    --lm.engine.max_total_tokens 2048 \
    --lm.engine.attention_backend triton
```

This example fits the BF16 model on a 24 GiB CUDA device with a short recording.
Adjust memory and context settings for the device, prompt, and recording length.
The LM admits one request at a time; concurrent requests expose admission queueing.
Run short, medium, and long caller recordings separately to compare workloads.
The profiler requires a torch build supporting
`torch._C._profiler._ExperimentalConfig(profile_all_threads=True)` because model
execution runs on worker threads. Unsupported builds fail explicitly. The CLI
enables `SGLANG_TORCH_PROFILER_PROFILE_ALL_THREADS=1` before starting workers;
ordinary serving keeps its existing profiler settings.

Each invocation creates a unique directory containing `report.json`, request event
JSONL files, and one compressed Chrome trace per worker process. Open decompressed
traces in Perfetto or another Chrome trace viewer. Co-located stages share a trace;
the `personaplex.*` scopes identify component ownership. Startup and a default one
request warmup are excluded from measurements. The two measured passes use the
same recording, voice, role prompt, seed, and greedy text/audio sampling.

The report provides:

- Unprofiled/profiled request latency, time to first audio chunk, real-time factor,
  throughput, and the profiler wall-time ratio.
- Exact full text and float32 waveform comparisons between the two passes, plus
  agreement between streamed audio and the terminal waveform. A mismatch fails
  the command while retaining the report.
- Separate CPU scope, CUDA launch, CUDA synchronization, GPU kernel, and GPU
  transfer durations for Temporal Transformer, text logits, Depformer, Mimi
  encoding/decoding, embeddings, H2D, and D2H. GPU memset is reported separately.
- GPU busy/idle interval unions, unattributed activity, missing component scopes,
  and kernel/transfer attribution coverage. Missing scopes remain null and fail
  the command; they are never reported as zero cost.
- Request timelines, preprocessing/Mimi compute intervals, stage admission waits,
  LM admission queueing, and stage-to-stage payload/stream hop latency.

CPU scopes describe launch-side elapsed time and may include synchronization;
their CUDA kernels can run after the scopes end. Kernels are attributed through
CUDA correlation and CPU external ids, including across worker threads. These
durations overlap and **must not be added into a disjoint latency total**. GPU idle
also includes host scheduling and other waiting, so it does not measure pure
launch overhead. Stage hop latency includes IPC, payload materialization, and
receiver scheduling; H2D/D2H scopes and GPU memcpy events describe device transfers.
Streaming Mimi scheduler waits are visible in hop and stage timing, without a
separate per-chunk scheduler queue metric. Use unprofiled latency for performance
claims and inspect the profiler overhead ratio before interpreting a bottleneck.
