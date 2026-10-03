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

The example targets a 24 GiB CUDA device with a short recording; adjust memory and
context settings for your workload. Requires a torch build supporting profiling
all worker threads. The LM admits one request at a time, so concurrency exposes
admission queueing.

After a default one-request warmup, the script runs the same input with greedy
sampling, first unprofiled and then profiled. Each run writes `report.json`, event
JSONL files, and compressed Chrome traces to a unique output directory. View
decompressed traces in Perfetto.

The report includes:

- CPU scopes, CUDA launch/synchronization, and GPU kernel/copy costs for Temporal
  Transformer, logits, Depformer, Mimi, embeddings, H2D, and D2H; GPU memset,
  busy/idle time, and attribution coverage are reported separately.
- Request latency, first-audio latency, RTF, throughput, and stage/queue/hop timing.
- Full text and float32 audio parity, streamed/terminal audio agreement, and
  profiler overhead. Output mismatches or missing component scopes fail the run.

**Timing metrics overlap; do not sum them.** Use unprofiled latency for performance
comparisons. GPU idle includes host scheduling/waiting; hop latency includes IPC
and receiver scheduling. Streaming Mimi has no separate per-chunk queue metric.
