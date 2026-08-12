# flashinfer-iket-lab

IKET (In-Kernel Event Tracing, `nvidia-cutlass-dsl` 4.7.0) applied to three classes of
FlashInfer CuTe-DSL kernels on B300 (SM103):

1. **Memory-bound op**: `flashinfer/cute_dsl/rmsnorm_fp4quant.py` — fused RMSNorm + NVFP4 quant.
2. **Compute-bound GEMM**: `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py` — the
   warp-specialized persistent NVFP4 block-scaled GEMM behind `mm_fp4(..., backend="cute-dsl")`.
3. **Fused communication kernel**: `flashinfer/cute_dsl/gemm_allreduce_two_shot.py` — GEMM with a
   fused two-shot multimem AllReduce epilogue, 8 GPUs over NVLink/NVLS, per-rank traces.

IKET emits named ranges from *inside* the kernel per warp (32 ns timer granularity), so producer
stalls, pipeline waits, and — in the comm case — cross-GPU skew become directly visible, which no
CUPTI-based tool shows. All numbers below are from the last (steady-state) launch in each trace.

## Environment

| item | value |
|---|---|
| GPU | 8× NVIDIA B300 SXM6 AC (SM103), driver 610.43.02 |
| container | flashinfer dev image, CUDA 13.2, torch 2.13.0.dev cu132, python 3.12 |
| DSL | `nvidia-cutlass-dsl[cu13]==4.7.0` (ships `run-iket`) |
| flashinfer | `main` @ `b8c21928` + `patches/iket-instrumentation.diff` |

Setup inside the container:

```bash
pip install "nvidia-cutlass-dsl[cu13]==4.7.0"
cd flashinfer && git apply /path/to/patches/iket-instrumentation.diff
pip install --no-build-isolation --no-deps -e .
```

## Results

### 1. RMSNorm + FP4 quant (M=2048, H=7168, bf16, block_size=16)

```
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/rmsnorm --clobber \
    profile --postprocess all -- python scripts/rmsnorm_iket_driver.py --rows 2048 --hidden 7168
```

Grid 2048×1 (1 row per CTA), 128 threads, 8192 warps. Per-warp phase means, steady-state launch:

| range | mean | share of 6.30 µs e2e |
|---|---|---|
| `sumsq_reduce` (smem read + x² + 4-warp smem reduction) | 2.10 µs | 33% |
| `quant_store` (re-read x from gmem, ×w, ×rstd, quant, pack, store) | 2.15 µs | 34% |
| `g2s_load` (cp.async G→S issue + wait) | 0.73 µs | 12% |
| `post_sync` (CTA barrier) | 0.10 µs | 2% |
| `setup` | 0.30 µs | 5% |

The kernel reads the input row twice (once via cp.async into smem for the sum of squares, once
from gmem inside the quant loop); IKET shows the two read phases costing the same ~2.1 µs each,
i.e. the second pass hits L2/L1 rather than HBM but pays the full FMA + convert + pack path.

### 2. NVFP4 block-scaled GEMM (M=N=4096, K=7168, `mm_fp4` cute-dsl backend)

```
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/gemm --clobber \
    profile --postprocess all -- python scripts/gemm_iket_driver.py --mnk 4096,4096,7168
```

148 CTAs (persistent), 512 output tiles, 28 k-blocks/tile, warp roles: 4 epilogue + 1 MMA + 1 TMA.
Steady-state launch, wall ≈ 40 µs, cos-sim vs bf16 reference 0.991:

| range (role) | count | mean | reading |
|---|---|---|---|
| `epi_acc_wait` (epilogue) | 2048 | 7.9 µs | epilogue warps spend ~83% of each tile **waiting for the accumulator** |
| `tma_acquire` (TMA) | 14336 | 201 ns | producer almost never stalls on free buffers |
| `tma_issue` (TMA) | 14336 | 43 ns | TMA issue is negligible |
| `mma_ab_wait` (MMA) | 7168 | 38 ns | MMA never waits on loads — memory has huge headroom |
| `mma_acc_acquire` (MMA) | 256 | 457 ns | accumulator pipeline rarely back-pressures |

Verdict at this shape: cleanly MMA-throughput-bound. The load pipeline (TMA) and the store side
could absorb a much larger K or coarser epilogue without moving the bottleneck.

### 3. GEMM + two-shot AllReduce, 8 GPUs (M=N=2048, K=4096, tf32→f32, NVLS multicast)

```
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/comm --clobber \
    profile --postprocess all -- torchrun --nproc-per-node 8 scripts/comm_iket_driver.py
```

256 output tiles, 148 persistent CTAs, 10 warps/CTA (4 epilogue + 1 MMA + 1 TMA + 4 AllReduce).
Reference check passed on all 8 ranks. Per-rank means from the 8 per-pid traces
(timestamps are trace-local; durations are comparable, absolute times across ranks are not):

| rank pid | kernel wall | `epi_ar_arrive` | `ar_bar_sync` | `ar_ldst` | `ar_flag_wait` |
|---|---|---|---|---|---|
| 0x86f | 276 µs | 5.3 µs | 15.3 µs | 69.7 µs | 1.1 µs |
| 0x870 | 583 µs | 113.5 µs | 50.9 µs | 68.8 µs | 1.1 µs |
| 0x871 | 525 µs | 80.3 µs | 44.2 µs | 69.3 µs | 1.1 µs |
| 0x872 | 206 µs | 80.6 µs | 7.2 µs | 70.0 µs | 1.1 µs |
| 0x873 | 541 µs | 100.1 µs | 46.0 µs | 69.8 µs | 1.1 µs |
| 0x874 | 331 µs | 5.8 µs | 21.8 µs | 69.4 µs | 1.1 µs |
| 0x875 | 207 µs | 77.5 µs | 7.4 µs | 69.2 µs | 1.1 µs |
| 0x876 | 215 µs | 61.6 µs | 8.4 µs | 68.4 µs | 1.1 µs |

Two observations that only an in-kernel view gives:

1. **The actual communication work is perfectly balanced**: `ar_ldst` (multimem `ld_reduce` +
   `st` sweep over this rank's 1/8 slice of each tile) is 68–70 µs on every rank.
2. **All the wall-time variance (206→583 µs) is absorbed at the sync sites** — the epilogue's
   per-tile multicast arrive (`epi_ar_arrive`, 5→113 µs) and the AR warps' intra-CTA barrier
   after the flag spin (`ar_bar_sync`, 7→51 µs). Ranks that enter the kernel early idle inside
   these waits until the stragglers' tiles land. The per-tile spin lock itself (`ar_flag_wait`)
   is uniformly ~1 µs — by the time warp 6 spins, the barrier upstream already absorbed the skew.

## Repo layout

- `patches/iket-instrumentation.diff` — all kernel edits, applies to flashinfer `b8c21928`
- `scripts/` — the three drivers + `analyze_iket_trace.py` (per-launch phase breakdown from the JSON traces)
- `traces/` — `.pftrace.gz` (open in https://ui.perfetto.dev/) and `.trace.json.gz` per experiment,
  including the pre-instrumented CUTLASS tutorial GEMM smoke test
- `results/` — analyzer outputs the tables above were built from

## Gotchas hit along the way

1. **`--h`-prefixed driver flags kill the run silently.** CuTe DSL's `diagnostic()` runs its own
   `argparse.parse_known_args()` at compile time; an app flag like `--h 4096` abbrev-matches
   `--help`, prints the DSL's help text, and exits the process before any kernel runs. The trace
   comes out empty. Use flag names that don't prefix `--help`.
2. **FlashInfer's CuTe-DSL disk cache defeats IKET.** Kernels reloaded from
   `~/.cache/flashinfer/.../cached_ops/*_cute_dsl/*.o` never JIT inside the profiled process, so
   they get no instrumentation and silently produce empty traces. Run everything with
   `FLASHINFER_CUTE_DSL_DISABLE_CACHE=1`.
3. **`mm_fp4(backend="cute-dsl")` on SM103 runs the *SM100* kernel class**
   (`dense_blockscaled_gemm_sm100.py`), not `dense_blockscaled_gemm_sm103.py`. Instrumenting the
   sm103 file (as this patch also does) yields nothing through the `mm_fp4` path. Verify which
   kernel actually launches (kernel name in the trace) before interpreting an empty result.
   The sm103-file instrumentation in the patch is included but was not exercised end-to-end.
4. **The TVM-FFI compile/launch path instruments fine** — `cute.compile(..., options="--enable-tvm-ffi")`
   plus TVM-FFI env-stream launches (rmsnorm) are captured like plain DSL launches.
5. **Multi-process torchrun workloads work out of the box** — the injection propagates to child
   processes and run-iket writes one trace per pid. The whole job runs twice (buffer-sizing pass +
   collection pass). Traces from different ranks are NOT on a shared timeline.
6. **IKET cannot run together with CUPTI tools** (Nsight Systems/Compute, cupti-python benchers) —
   separate runs.

## Reproduce

```bash
# container: any CUDA 13.x flashinfer dev image on Blackwell (SM100/103 for these three kernels)
pip install "nvidia-cutlass-dsl[cu13]==4.7.0"
git clone https://github.com/flashinfer-ai/flashinfer && cd flashinfer && git checkout b8c21928
git apply ../patches/iket-instrumentation.diff
pip install --no-build-isolation --no-deps -e . && cd ..

FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/rmsnorm --clobber profile --postprocess all -- \
    python scripts/rmsnorm_iket_driver.py --rows 2048 --hidden 7168
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/gemm --clobber profile --postprocess all -- \
    python scripts/gemm_iket_driver.py --mnk 4096,4096,7168
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/comm --clobber profile --postprocess all -- \
    torchrun --nproc-per-node 8 scripts/comm_iket_driver.py   # needs 8 GPUs + NVLS

python scripts/analyze_iket_trace.py --launch -1 out/*/iket_pid_*.trace.json
```

IKET is experimental (API/output format may change); see the CUTLASS "IKET Profiling" guide.
