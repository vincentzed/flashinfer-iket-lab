# flashinfer-iket-lab

This repository documents, step by step, how we used IKET to profile four kinds of FlashInfer
CuTe-DSL kernels on an 8x NVIDIA B300 (SM103) machine. Every step below shows the exact command
we ran and the raw output it printed. The full uncut logs are in `logs/`. The traces are in
`traces/`. The kernel changes are one patch file in `patches/`.

**What is IKET?** IKET (In-Kernel Event Tracing) is an experimental profiler that ships with
`nvidia-cutlass-dsl` 4.7.0. You add named markers and ranges *inside* a CuTe-DSL kernel
(`cute.experimental.iket.range_push("name")` / `range_pop()`), run your program under the
`run-iket` tool, and get a per-warp timeline with 32 ns resolution. This shows things CUPTI-based
tools (Nsight, torch profiler) cannot show: how long each warp spent inside a spin-wait, a
pipeline stall, or a barrier, and — for multi-GPU kernels — how cross-GPU skew is absorbed
inside the kernel. IKET only works on CuTe-DSL kernels. It cannot profile CUDA C++, Triton, or
cuTile kernels.

**What we profiled:**

1. A state-of-the-art memory-bound op: fused RMSNorm + NVFP4 quantization (`rmsnorm_fp4quant`).
2. A compute-bound GEMM: the NVFP4 block-scaled persistent GEMM behind `mm_fp4(..., backend="cute-dsl")`.
3. A fused communication kernel: GEMM with a two-shot multimem AllReduce in its epilogue, on 8 GPUs.
4. The three MNNVL AllReduce protocols (LL = low latency, BT = balanced, HT = high throughput),
   compared head-to-head at real shapes with a Latin-square A/B design.

A survey of every other communication kernel in flashinfer, and why these were chosen, is in
[COMM_KERNEL_SURVEY.md](COMM_KERNEL_SURVEY.md).

## Hardware and software

| item | value |
|---|---|
| GPUs | 8x NVIDIA B300 SXM6 AC (SM103), driver 610.43.02, NVLink + NVLS multicast |
| container image | a flashinfer dev image: CUDA 13.2, torch 2.13.0.dev cu132, python 3.12 |
| DSL | `nvidia-cutlass-dsl[cu13]==4.7.0` (this version ships the `run-iket` tool) |
| flashinfer | `main` at commit `b8c21928`, plus `patches/iket-instrumentation.diff` |

---

## Step 1: start a container and make the GPUs visible

We started a fresh container from our existing flashinfer dev image. The image was built before
driver 610 was installed on the host, so two box-specific fixes were needed: copy the host's
NVML library into the container (otherwise `nvidia-smi` fails), and change the container user's
uid to match the host user (otherwise the bind-mounted home directory is unreadable).

```bash
docker run -d --name flashinfer-iket-dev --gpus all --network=host --shm-size=64g \
    --cap-add=SYS_PTRACE --ipc=host -v /home/brayden:/home/brayden \
    --workdir /home/brayden/iket-lab flashinfer-cu132-dev:local sleep infinity

docker cp /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.610.43.02 flashinfer-iket-dev:/usr/lib/x86_64-linux-gnu/
docker exec -u root flashinfer-iket-dev bash -c 'cd /usr/lib/x86_64-linux-gnu && \
    ln -sf libnvidia-ml.so.610.43.02 libnvidia-ml.so.1 && ln -sf libnvidia-ml.so.1 libnvidia-ml.so && ldconfig'
```

Raw output of the verification:

```
GPU 0: NVIDIA B300 SXM6 AC (UUID: GPU-daaa8d02-04fc-d666-a488-8eb3721df133)
GPU 1: NVIDIA B300 SXM6 AC (UUID: GPU-d1d60fa6-6734-0390-5487-e6f8b4b572a9)
```

## Step 2: upgrade nvidia-cutlass-dsl to 4.7.0

The image shipped cutedsl 4.5.1, which has no IKET. Version 4.7.0 is the first we used that
ships `run-iket`.

```bash
pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.7.0" && run-iket --help
```

Raw output (pip tail plus the help text):

```
    Found existing installation: nvidia-cutlass-dsl 4.5.1
    Uninstalling nvidia-cutlass-dsl-4.5.1:
      Successfully uninstalled nvidia-cutlass-dsl-4.5.1

Successfully installed nvidia-cuda-nvdisasm-13.3.73 nvidia-cutlass-dsl-4.7.0 nvidia-cutlass-dsl-libs-base-4.7.0 nvidia-cutlass-dsl-libs-core-4.7.0 nvidia-cutlass-dsl-libs-cu12-4.7.0 nvidia-cutlass-dsl-libs-cu13-4.7.0 protobuf-6.33.6
usage: run-iket [-h] [--use-injection-lib USE_INJECTION_LIB] [--skip-run]
                [--context-buffer-size CONTEXT_BUFFER_SIZE]
                [--log-level {error,warn,info,debug,trace}]
                [--working-dir WORKING_DIR] [--output-dir OUTPUT_DIR]
                [--clobber]
                {profile,postprocess} ...

positional arguments:
  {profile,postprocess}
    profile             run In-Kernel-Event-Tracing
    postprocess         post-process an existing IKET run directory to
                        generate traces
```

## Step 3: install flashinfer from source and apply the instrumentation patch

We checked out flashinfer `main` at `b8c21928`, initialized submodules, applied our patch, and
installed it editable. `--no-deps` protects the image's torch install.

```bash
git clone https://github.com/flashinfer-ai/flashinfer && cd flashinfer
git checkout b8c21928 && git submodule update --init --recursive
git apply ../patches/iket-instrumentation.diff
pip install --no-build-isolation --no-deps -e .
python -c "import flashinfer; print(flashinfer.__version__)"
```

Raw output of the install and import check:

```
Successfully built flashinfer-python
Installing collected packages: flashinfer-python
Successfully installed flashinfer-python-0.6.18
0.6.18
```

## Step 4: smoke test with NVIDIA's own pre-instrumented tutorial

Before touching flashinfer kernels, we ran the CUTLASS tutorial GEMM that NVIDIA ships already
instrumented (`fp16_gemm_4_iket.py`). This validates the whole IKET pipeline on SM103.

```bash
CUDA_VISIBLE_DEVICES=7 run-iket --output-dir out/smoke --clobber profile --postprocess all -- \
    python /path/to/cutlass/examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py --mnk 512,1024,64
```

Raw output, last lines (full file: `logs/smoke_run.log`):

```
Verifying reference result...
PASS
[run-iket] Dumped perfetto trace to /home/brayden/iket-lab/out/smoke/iket_pid_0xa61.pftrace
[run-iket] Dumped json trace to /home/brayden/iket-lab/out/smoke/iket_pid_0xa61.trace.json
[run-iket] Dumped compressed trace to /home/brayden/iket-lab/out/smoke/iket_pid_0xa61.pftrace.gz
[run-iket] Dumped perfetto HTML viewer to /home/brayden/iket-lab/out/smoke/iket_pid_0xa61.html
```

The trace summary (`results/smoke_summary.txt`, produced by `scripts/analyze_iket_trace.py`)
shows 28 named ranges across the tutorial's warp roles, which confirms markers, ranges, JSON,
and Perfetto output all work on this machine.

---

## Step 5: experiment 1 — memory-bound op (fused RMSNorm + FP4 quant)

**What we instrumented.** We added six ranges to the kernel in
`flashinfer/cute_dsl/rmsnorm_fp4quant.py`: `setup`, `g2s_load` (the cp.async copy of the input
row into shared memory), `sumsq_reduce` (sum of squares plus the cross-warp reduction),
`post_sync` (the barrier after the reduction), `quant_store` (the fused multiply, quantize,
pack, and store loop), and a whole-kernel range `kernel_e2e`.

**The shape.** 2048 tokens by hidden size 7168 in bf16. 7168 is the hidden size of
DeepSeek-V3/R1; 2048 tokens is a realistic chunked-prefill batch.

**The command and its raw output** (full file: `logs/rmsnorm_run.log`):

```bash
CUDA_VISIBLE_DEVICES=7 FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/rmsnorm \
    --clobber profile --postprocess all -- python scripts/rmsnorm_iket_driver.py --rows 2048 --hidden 7168
```

```
[run-iket] ============= Dry run application first to collect some info... =============
ran m=2048 h=7168 dtype=bfloat16 iters=2
y_fp4 (2048, 3584) torch.float4_e2m1fn_x2, scales (2048, 448) torch.float8_e4m3fn
ref row-norm max |y|: 9.707 (sanity only)
[run-iket] Auto-computed context buffer size: 0.00 GB (2,260,992 bytes)
[run-iket] ============= Run application for real profiling... =============
ran m=2048 h=7168 dtype=bfloat16 iters=2
y_fp4 (2048, 3584) torch.float4_e2m1fn_x2, scales (2048, 448) torch.float8_e4m3fn
ref row-norm max |y|: 9.707 (sanity only)
[run-iket] Dumped perfetto trace to /home/brayden/iket-lab/out/rmsnorm/iket_pid_0xbdf.pftrace
[run-iket] Dumped json trace to /home/brayden/iket-lab/out/rmsnorm/iket_pid_0xbdf.trace.json
```

**The analysis command and its raw output** (also saved as `results/rmsnorm_summary.txt`):

```bash
python scripts/analyze_iket_trace.py --launch -1 out/rmsnorm/iket_pid_0xbdf.trace.json
```

```
=== out/rmsnorm/iket_pid_0xbdf.trace.json (3 instrumented launches) ===

[eager] kernel=kernel_cutlass_kernel_flashinfercute_dslrmsnorm_fp4quantRMSNormFP4Quan
  grid=(2048,1,1) block=(128,1,1) warps=8192 markers=0 wall(first-warp-start -> last-warp-end)=9.2 us
  range               count   total_us    mean_ns     p50_ns     p99_ns     max_ns
  kernel_e2e           8192    51862.9     6330.9       6272       8320       8608
  quant_store          8192    17679.0     2158.1       2112       3008       3328
  sumsq_reduce         8192    17292.2     2110.9       1824       3584       3936
  g2s_load             8192     5994.2      731.7        640       1920       3168
  setup                8192     2428.3      296.4        192       1600       2368
  post_sync            8192      833.7      101.8         96        384        736
```

**What this says, in plain words.** Each warp spends about 6.3 microseconds in the kernel. The
two big phases cost the same: computing the sum of squares takes 2.11 microseconds, and the
quantize-and-store loop takes 2.16 microseconds. This is because the kernel reads the input row
twice: once through shared memory for the sum of squares, and a second time directly from global
memory inside the quantization loop. The second read mostly hits cache, but the fused multiply,
maximum-search, FP4 conversion, and packing work makes the second pass as expensive as the first.

## Step 6: experiment 2 — compute-bound GEMM (NVFP4 block-scaled, `mm_fp4` cute-dsl backend)

**What we instrumented.** The kernel is warp-specialized: 4 epilogue warps, 1 MMA warp, and
1 TMA (load) warp per CTA. We added 13 ranges to
`flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py`: per-role "main" and per-tile ranges,
`tma_acquire` (waiting for a free buffer) and `tma_issue` (issuing the loads) in the load loop,
`mma_ab_wait` (MMA waiting for data) and `mma_acc_acquire` (MMA waiting for the accumulator),
and `epi_acc_wait` (epilogue waiting for the accumulator).

**The shape.** M=N=4096, K=7168, NVFP4 with block size 16. This is a realistic large dense
GEMM for NVFP4-quantized serving (7168 is the DeepSeek-V3 hidden size).

**The command and its raw output** (full file: `logs/gemm_run.log`):

```bash
CUDA_VISIBLE_DEVICES=7 FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/gemm \
    --clobber profile --postprocess all -- python scripts/gemm_iket_driver.py --mnk 4096,4096,7168
```

```
[run-iket] ============= Dry run application first to collect some info... =============
ran mnk=(4096,4096,7168) iters=2 cos_sim=0.99097
[run-iket] Auto-computed context buffer size: 0.01 GB (6,911,481 bytes)
[run-iket] ============= Run application for real profiling... =============
ran mnk=(4096,4096,7168) iters=2 cos_sim=0.99097
[run-iket] Dumped perfetto trace to /home/brayden/iket-lab/out/gemm/iket_pid_0xc73.pftrace
[run-iket] Dumped json trace to /home/brayden/iket-lab/out/gemm/iket_pid_0xc73.trace.json
```

(`cos_sim=0.99097` is the cosine similarity of the FP4 GEMM result against a bf16 torch.mm
reference; the driver asserts it is above 0.97.)

**The analysis command and its raw output** (also saved as `results/gemm_summary.txt`):

```bash
python scripts/analyze_iket_trace.py --launch -1 out/gemm/iket_pid_0xc73.trace.json
```

```
=== out/gemm/iket_pid_0xc73.trace.json (3 instrumented launches) ===

[eager] kernel=kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm10
  grid=(2,1,74) block=(192,1,1) warps=888 markers=0 wall(first-warp-start -> last-warp-end)=39.9 us
  range               count   total_us    mean_ns     p50_ns     p99_ns     max_ns
  kernel_e2e            888    27640.3    31126.4      29920      39040      39072
  epi_main              592    19592.3    33095.0      29216      38208      38272
  epi_tile             2048    19523.1     9532.8       9280      10880      11072
  epi_acc_wait         2048    16254.0     7936.5       7648       9216       9312
  tma_main              148     4719.9    31891.5      27872      37024      37024
  tma_tile              512     4510.3     8809.2       9088       9472       9568
  tma_acquire         14336     2887.1      201.4        192        736        800
  mma_main              148     2670.6    18044.5      27424      37088      37120
  mma_tile              512     2594.5     5067.4       8416       9536       9568
  tma_issue           14336      614.8       42.9         32        128        384
  prologue              888      293.2      330.2        320        384        384
  mma_ab_wait          7168      270.3       37.4         32        256        576
  mma_acc_acquire       256      117.0      457.1        608        704        768
```

**What this says, in plain words.** The kernel runs 512 output tiles on 148 persistent CTAs,
with 28 load steps per tile. The epilogue warps spend 7.9 of their 9.5 microseconds per tile
just waiting for the accumulator (`epi_acc_wait`). Meanwhile the MMA warp almost never waits
for data (`mma_ab_wait` is 37 nanoseconds on average) and the TMA warp almost never stalls
(`tma_acquire` is 201 nanoseconds, `tma_issue` is 43 nanoseconds). Conclusion: at this shape
the kernel is limited purely by tensor-core (MMA) throughput. The memory pipeline has large
headroom.

## Step 7: experiment 3 — fused communication kernel (GEMM + two-shot AllReduce, 8 GPUs)

**What we instrumented.** `flashinfer/cute_dsl/gemm_allreduce_two_shot.py` runs GEMM warps plus
4 dedicated AllReduce warps per CTA. We added: `epi_ar_arrive` (the epilogue's per-tile
multicast "my data is ready" signal), `ar_flag_wait` (spin-waiting until all 8 ranks have
signaled a tile), `ar_bar_sync` (the barrier between the 4 AR warps), `ar_ldst` (the actual
multimem ld_reduce + multicast store sweep), and `ar_final_bar` (the end-of-kernel system
barrier).

**The shape.** M=N=2048, K=4096, tf32 inputs, f32 output, 8 ranks — the configuration of the
upstream test `test_cute_dsl_gemm_allreduce_two_shot.py`. The kernel requires world_size=8 when
the allreduce is enabled.

**The command and its raw output** (full file: `logs/comm_run.log`; the run prints the
config from all 8 ranks and a reference check passes on each):

```bash
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/comm --clobber profile --postprocess all -- \
    torchrun --nproc-per-node 8 scripts/comm_iket_driver.py
```

```
Running Blackwell Persistent Dense GEMM test with:
mnkl: (2048, 2048, 4096, 1)
AB dtype: TFloat32, C dtype: Float32, Acc dtype: Float32
Mma Tiler (M, N): (128, 128), Cluster Shape (M, N): (1, 1)
Fused AllReduce Op: two_shot
...
rank 3: gemm+two_shot allreduce OK    (8 of these lines, one per rank, per pass)
```

run-iket writes one trace per rank process. **The per-rank comparison command and its raw
output** (also saved as `results/comm_skew_table.txt`):

```bash
python scripts/comm_skew_table.py out/comm/iket_pid_*.trace.json
```

```
pid        wall_us epi_arrive  bar_sync     ldst flag_wait final_bar  (mean us, last launch)
0x829    (no instrumented launches - torchrun parent process)
0x86f        275.7        5.3      15.3     69.7      1.12       4.7
0x870        583.1      113.5      50.9     68.8      1.13       4.7
0x871        525.4       80.3      44.2     69.3      1.09       4.3
0x872        205.9       80.6       7.2     70.0      1.11       4.9
0x873        540.6      100.1      46.0     69.8      1.11       4.7
0x874        331.2        5.8      21.8     69.4      1.14       4.4
0x875        207.0       77.5       7.4     69.2      1.13       4.9
0x876        214.5       61.6       8.4     68.4      1.12       4.8
```

**What this says, in plain words.** Two things stand out. First, the actual communication work
(`ldst`, the multimem load-reduce and store sweep) takes 68 to 70 microseconds on every rank —
the work itself is perfectly balanced. Second, the total time each rank spends inside the kernel
varies enormously, from 206 to 583 microseconds. All of that variation sits inside the waiting
phases (`epi_arrive` and `bar_sync`). Ranks whose kernel starts early idle inside these waits
until the slowest rank's tiles arrive. This is launch skew being absorbed at specific,
identifiable sync points — something you cannot see from a CUPTI timeline, because from the
outside every rank's kernel just looks "long". One caveat: each rank's trace has its own
timebase, so durations are comparable across ranks but absolute start times are not.

## Step 8: experiment 4 — LL vs BT vs HT AllReduce protocols (Latin-square A/B at real shapes)

**Background.** flashinfer's newest CuTe-DSL communication backend
(`flashinfer/comm/mnnvl_cutedsl/`, last touched two days before this experiment) implements the
fused AllReduce + residual-add + RMSNorm operation three times, with three protocols: LL
("low latency", Lamport-flag based), BT ("balanced"), and HT ("high throughput"). The public
router picks a protocol from the token count. We forced each protocol with its `*_ONLY_CONFIG`
and compared them head-to-head.

**What we instrumented.** The LL Lamport kernel got full phase ranges (`ll_preload`,
`ll_pdl_wait`, `ll_spin_reduce` — the Lamport spin that waits for all ranks' contributions —
`ll_sentinel_clear`, `ll_rms_reduce`, `ll_norm_store`, plus `ll_lamport_e2e`). The BT
owner-reduce kernel got a `bt_spin_reduce` range around its spin loop plus markers; every other
kernel in the three pipelines got a marker so IKET records its warp lifetimes (LL = publish +
lamport kernels, BT = publish + owner-reduce + rmsnorm-materialize kernels, HT = one kernel).

**Real shapes only.** The upstream presets are tuned for exactly one static configuration:
hidden size 8192, TP 8 (or 16), bf16, on GB300-class hardware — which is precisely our box
(8x B300, TP8). So the realistic axis is the token count m. We used
m = 8, 32 (decode batches), 256, 1024 (mixed/medium batches), 4096, 8192 (prefill chunks).
These bracket the router's own crossover points (the default config switches LL to BT at about
m=52 and BT to HT at m=1024). One honest limitation surfaced by the run: `BT_ONLY_CONFIG`'s
routes stop at m=1024, so for m=4096 and m=8192 the comparison is LL vs HT only.

**The Latin square.** For each m, the protocols run in rotated orders — (LL,BT,HT),
(BT,HT,LL), (HT,LL,BT) — so every protocol appears exactly once in every position. This cancels
order effects (clock ramp-up, cache state, which protocol ran right after its compile). Above
m=1024, two balanced 2x2 rounds of (LL,HT)/(HT,LL) are used. One warmup execution per
(m, protocol) runs before each square and is excluded from the statistics. Every warmup is also
checked numerically against a torch reference (all-gather, sum, residual add, RMSNorm).

**The command and its raw output** (full file: `logs/mnnvl_run.log`):

```bash
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/mnnvl --clobber profile --postprocess all -- \
    torchrun --nproc-per-node 8 scripts/mnnvl_protocols_iket_driver.py
```

<!--MNNVL_MANIFEST-->

**The analysis command and its raw output** (also saved as `results/mnnvl_protocol_report.txt`):

```bash
python scripts/mnnvl_protocol_report.py logs/mnnvl_run.log out/mnnvl/iket_pid_*.trace.json
```

<!--MNNVL_REPORT-->

<!--MNNVL_READING-->

## Step 9: how to look at the traces yourself

Every experiment directory in `traces/` contains one `.pftrace.gz` per process. Open
https://ui.perfetto.dev/ in a browser and load the file (Perfetto reads .gz directly). Expand
the tracks: they are grouped by GPU location (SM, CTA, warp), and the named ranges from the
patch appear per warp. Use the W/A/S/D keys to zoom and pan. The `.trace.json.gz` files contain
the same events for scripted analysis; `scripts/analyze_iket_trace.py` shows how to read them.

---

## The mistakes we made, so you do not repeat them

1. **A driver flag named `--h` silently kills the run.** CuTe DSL runs its own
   `argparse.parse_known_args()` when it compiles a kernel. An application flag such as
   `--h 4096` abbreviation-matches argparse's built-in `--help`, so the DSL's parser prints a
   help text and exits the process before any kernel runs. The trace comes out empty with no
   error. Do not use flag names that are prefixes of `--help`.
2. **FlashInfer's CuTe-DSL disk cache silently defeats IKET.** FlashInfer caches compiled
   CuTe-DSL kernels as `.o` files and reloads them on later runs. A reloaded kernel never
   JIT-compiles inside the profiled process, so run-iket cannot instrument it, and the trace
   comes out empty. Always set `FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` when profiling flashinfer.
3. **`mm_fp4(backend="cute-dsl")` on SM103 does not run the SM103 kernel file.** It runs the
   `Sm100BlockScaledPersistentDenseGemmKernel` class from `dense_blockscaled_gemm_sm100.py`,
   even on SM103 hardware. We instrumented `dense_blockscaled_gemm_sm103.py` first and got an
   empty trace. Check the kernel name inside the trace before trusting your instrumentation
   placement. (The patch still contains the sm103-file instrumentation, but it was never
   exercised end to end.)
4. **`BT_ONLY_CONFIG` cannot serve more than 1024 tokens.** Building its workspace with a larger
   capacity raises `ValueError: Finalize routes do not cover the requested workspace capacity`.
   This is an upstream routing bound, not a bug in the experiment.
5. **IKET cannot run together with CUPTI tools.** Do not combine run-iket with Nsight Systems,
   Nsight Compute, or cupti-python benchmarking in the same run.
6. **Multi-process notes.** torchrun works out of the box (the injection reaches child
   processes), each rank gets its own trace file, the whole job runs twice (a buffer-sizing
   pass, then the collection pass), and rank traces are not on a shared timeline.

## Repository layout

- `patches/iket-instrumentation.diff` — every kernel edit, applies onto flashinfer `b8c21928`
- `scripts/` — four drivers plus two analysis scripts
- `logs/` — full raw stdout/stderr of every profiled run
- `traces/` — Perfetto traces (`.pftrace.gz`) and JSON traces (`.trace.json.gz`) per experiment
- `results/` — raw analyzer outputs quoted in this README
- `COMM_KERNEL_SURVEY.md` — all communication kernels in flashinfer, classified by IKET-ability

IKET is experimental; its API and output format may change. See the CUTLASS documentation page
"IKET Profiling (In-Kernel Event Tracing)".
