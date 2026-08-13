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
5. **The centerpiece: the CuTeDSL MegaMoE megakernel at DeepSeek-V4-Pro shapes** — profiled
   with IKET, hillclimbed under a bitwise-exactness rule, benchmarked head-to-head against
   DeepSeek's DeepGEMM megamoe, and contrasted with what Nsight Compute can (and cannot) see.
   This is Step 9 below.

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

**The shape and dtypes.** M=N=2048, K=4096, 8 ranks, with bf16 A/B inputs, bf16 output, and
f32 accumulation — the real tensor-parallel serving dtype. (The upstream test uses tf32/f32;
we changed only the dtypes. The bf16 output exercises the `multimem_ld_reduce_8xbf16` path in
the allreduce epilogue.) The kernel requires world_size=8 when the allreduce is enabled.

**The command and its raw output** (full file: `logs/comm_run.log`; the run prints the
config from all 8 ranks and a reference check passes on each):

```bash
FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/comm --clobber profile --postprocess all -- \
    torchrun --nproc-per-node 8 scripts/comm_iket_driver.py
```

```
Running Blackwell Persistent Dense GEMM test with:
mnkl: (2048, 2048, 4096, 1)
AB dtype: BFloat16, C dtype: BFloat16, Acc dtype: Float32
Mma Tiler (M, N): (128, 128), Cluster Shape (M, N): (1, 1)
Fused AllReduce Op: two_shot
...
rank N: gemm+two_shot allreduce OK    (16 lines total: 8 ranks x 2 passes, reference check on each)
```

run-iket writes one trace per rank process. **The per-rank comparison command and its raw
output** (also saved as `results/comm_skew_table.txt`):

```bash
python scripts/comm_skew_table.py out/comm/iket_pid_*.trace.json
```

```
pid        wall_us epi_arrive  bar_sync     ldst flag_wait final_bar  (mean us, last launch)
0x134e   (no instrumented launches - torchrun parent process)
0x1394       552.2       88.9      49.6     60.0      1.13       3.5
0x1395       384.6        5.5      30.3     60.2      1.11       3.5
0x1396       187.8       77.2       7.8     58.2      1.10       3.8
0x1397       351.4        5.3      26.4     59.9      1.15       3.7
0x1398       240.6      103.3      12.6     65.3      1.20       3.8
0x1399       490.8       53.6      42.5     60.4      1.09       3.7
0x139a       438.5       23.9      36.5     60.6      1.15       3.3
0x139b       785.6       85.6      76.7     59.8      1.09       3.6
```

**What this says, in plain words.** Two things stand out. First, the actual communication work
(`ldst`, the multimem load-reduce and multicast-store sweep, here on the bf16 wire) takes 58 to
65 microseconds on every rank — the work itself is balanced. Second, the total time each rank
spends inside the kernel varies enormously, from 188 to 786 microseconds. Almost all of that
variation sits inside the waiting phases (`epi_arrive` and `bar_sync`). Ranks whose kernel
starts early idle inside these waits until the slowest rank's tiles arrive. This is launch skew
being absorbed at specific, identifiable sync points — something a CUPTI timeline cannot show,
because from the outside every rank's kernel just looks "long". One caveat: each rank's trace
has its own timebase, so durations are comparable across ranks but absolute start times are not.

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

Raw output, first manifest lines (every warmup is numerically checked against a torch
all-gather + sum + residual + RMSNorm reference; `norm_cos` is the cosine similarity):

```
MANIFEST exec=0 m=8 phase=warmup pos=- proto=ll norm_cos=0.999997 prenorm_max_diff=0.0120
MANIFEST exec=1 m=8 phase=warmup pos=- proto=bt norm_cos=0.999997 prenorm_max_diff=0.0120
MANIFEST exec=2 m=8 phase=warmup pos=- proto=ht norm_cos=0.999997 prenorm_max_diff=0.0149
MANIFEST exec=3 m=8 phase=latin order=0 pos=0 proto=ll
MANIFEST exec=4 m=8 phase=latin order=0 pos=1 proto=bt
MANIFEST exec=5 m=8 phase=latin order=0 pos=2 proto=ht
...
done: 68 executions across m=[8, 32, 256, 1024, 4096, 8192]
```

**The analysis command and its raw output** (also saved as `results/mnnvl_protocol_report.txt`):

```bash
python scripts/mnnvl_protocol_report.py logs/mnnvl_run.log out/mnnvl/iket_pid_*.trace.json
```

```
== span per (m, protocol), latin executions pooled across ranks ==
     m  proto  reps   mean_us   min_us   max_us
     8     bt     3     144.4    120.4    163.8
     8     ht     3      20.2     18.2     21.3
     8     ll     3      96.8     70.0    113.8
    32     bt     3     170.8    154.0    186.0
    32     ht     3      20.2     16.3     27.6
    32     ll     3      76.1     71.9     78.6
   256     bt     3     255.0    146.7    457.1
   256     ht     3      61.9     33.5     81.6
   256     ll     3     103.1     95.3    109.0
  1024     bt     3     628.5    189.5   1258.2
  1024     ht     3     144.3     61.3    302.6
  1024     ll     3     300.7    291.2    315.5
  4096     ht     4     187.6    174.5    218.1
  4096     ll     4     985.8    837.4   1041.0
  8192     ht     4     344.9    318.6    374.7
  8192     ll     4    1740.9   1247.3   2030.8

== LL lamport kernel phase means (ns per warp) ==
     m    lamport_e2e     norm_store       pdl_wait        preload     rms_reduce sentinel_clear    spin_reduce
     8           2729             38              8            266            801            160           1365
    32           3012             40             10            283           1078            160           1325
   256           6126             56             16            581           2608            193           2403
  1024           6834             53             21            999           2660            147           2743
  4096           9228             53             17           1079           2815            136           4932
```

(The full output including the Latin-square position table and the BT spin distribution is in
`results/mnnvl_protocol_report.txt`. This report was generated from the rank traces that had
finished postprocessing; each rank's trace contains every execution, so one trace already
covers the whole Latin square.)

**What this says, in plain words.** On this machine (one node, 8x B300, NVLS multicast), the
high-throughput (HT) protocol wins at every token count we tested — including the smallest.
At m=8, HT finishes in about 20 microseconds while LL takes about 97 and BT about 144. At
m=8192, HT is about 5x faster than LL (345 vs 1741 microseconds). That is surprising, because
the shipped default router sends token counts up to about 50 to LL. Two honest caveats. First,
these presets were tuned for multi-node GB300 NVL72 systems, where LL's Lamport design fights
real fabric latency; a single node is not that environment. Second, our LL kernel carries about
14 IKET events per warp versus 1 for HT, so LL pays more instrumentation overhead — but that is
sub-microsecond per warp and cannot explain a 4-9x gap. The phase table explains the mechanism:
LL launches one CTA row per token (grid scales with m), and its per-warp `spin_reduce` and
`rms_reduce` phases grow with m, while HT keeps a fixed persistent grid. The Latin-square
position table shows no consistent first-position penalty, so the ranking is not an ordering
artifact; BT's spread (its `bt_spin_reduce` p-max hits 247 microseconds at m=1024) is real
variance in its spin phase, not measurement noise.

## Step 9: the centerpiece — CuTeDSL MegaMoE at DeepSeek-V4-Pro shapes

**Why this kernel.** A megakernel is the case where every classical profiler is blind by
construction: expert-parallel dispatch over NVLink, the grouped FC1 GEMM, SwiGLU, FC2, and the
cross-rank combine all execute inside one persistent kernel launch, so a CUPTI timeline shows
one long rectangle per layer. FlashInfer's `moe_ep` "mega" path ships two CuTe-DSL megakernels
(NVFP4 and MXFP8, from the NVIDIA kernel team, PR #3852/#3980/#4079). And here is the detail
that makes this repository's point better than anything we wrote ourselves: **the kernel team
ships these megakernels with IKET spans already in the device code** — `Dispatch_Prep`,
`Dispatch_Barrier`, `Pull.TMA_NVLink_Roundtrip`, `Pull.Arrival_Atomic`, `tma_weight_fc1`,
`tma_token_fc1_wait`, `mma_fc1`, `fc1_epi`, `mma_fc2`, `token_back`, `Kernel_Tail`, and more,
behind a no-op compatibility shim (`src/src/iket_compat.py`). With `nvidia-cutlass-dsl` 4.7.0
installed, the real IKET dialect loads and every span lights up under `run-iket`. In-kernel
tracing is not a research toy: it is how the people who write megakernels debug them.

**Real shapes.** From DeepSeek-V4-Pro's `config.json` (local HF cache,
`deepseek-ai/DeepSeek-V4-Pro`): hidden_size 7168, moe_intermediate_size 3072, 384 routed
experts, top-k 6, `expert_dtype = fp4` — which makes the NVFP4 CuTeDSL megamoe literally this
model's serving path. EP4 (4 GPUs) is the upstream-validated deployment geometry
(DeepSeek-V4-Flash e2e ran 4x GB200 TP4/EP4); the hillclimb below runs at world_size 8
(EP8, 48 local experts per rank) as required.

We also applied the CuTe-DSL 4.7.0 staging fix from open flashinfer PR #4449 (a verbatim
re-sync of `src/src/inputs_process.py`, which reworks the fused bf16→quant staging that could
misalign under 4.7.0) before trusting any numbers.

**Correctness first.** The driver (`scripts/megamoe_iket_driver.py`) reuses the upstream
multirank test helpers and checks the kernel against a pure-torch oracle over the global
384-expert set. Raw output at EP8:

```
ORACLE rank=4 rel_l2=0.002623 max|d|=512
ORACLE rank=3 rel_l2=0.002612 max|d|=512
ORACLE rank=2 rel_l2=0.002618 max|d|=512
ORACLE rank=6 rel_l2=0.002627 max|d|=512
ORACLE rank=0 rel_l2=0.002621 max|d|=512
ORACLE rank=1 rel_l2=0.002627 max|d|=512
ORACLE rank=7 rel_l2=0.002621 max|d|=512
ORACLE rank=5 rel_l2=0.002621 max|d|=512
```

(rel_l2 0.0026 is NVFP4 round-to-nearest noise; the outputs' magnitude with random unscaled
weights makes max|d|=512 ≈ 0.3% of the output range, matching upstream's own oracle bands.)

**What IKET shows inside the megakernel at decode shape** (m=128 tokens/rank, EP4, one rank's
steady-state launch; full output in `results/megamoe_m128_summary.txt`):

```
  range               count   total_us    mean_ns     p50_ns     p99_ns     max_ns
  tma_token_fc1        4608    54728.9    11876.9      11456      24832      25728
  tma_weight_fc1       4608    54711.4    11873.1      11424      24672      25568
  mma_fc1              4608    27641.3     5998.5       7488      26848      29344
  tma_token_fc2        5376    25328.5     4711.4       4672       7552      15872
  tma_weight_fc2       5376    25289.5     4704.2       4672       7488      15840
  fc1_epi             18432    22183.7     1203.5       1120       2528       3488
  produce_tile_id     10132    21821.8     2153.7       1728      14368      14784
  mma_fc2              5376    12740.1     2369.8       1984       6688      14880
  fc2_epi             21504     9533.7      443.3        384       1408       3456
  tma_token_fc1_wait      592     2471.8     4175.3        672      15520      15648
  Dispatch_Barrier        1        8.8     8832.0       8832       8832       8832
  ...
```

Reading: at decode batch (128 tokens x top-6 across 384 experts ≈ 16 tokens per expert), the
fc1 **weight loads run at twice the MMA time** (`tma_weight_fc1` ~11.9 µs vs `mma_fc1` ~6.0 µs
per task) — the kernel is weight-bandwidth bound, exactly as MoE decode theory predicts. The
cross-rank dispatch machinery is nearly free at this size (`Dispatch_Barrier` microseconds,
`Pull.*` spans tiny). This diagnosis directed the whole hillclimb: at decode, no combine-wire
or scheduling knob can create bandwidth, but tile shape can trim wasted work; at prefill, the
growing term is combine traffic, which the combine wire dtype directly controls.

**The hillclimb, under a bitwise-exactness rule.** Every candidate that claims to speed up the
baseline had to reproduce the baseline's output bit-for-bit (`torch.equal`, per rank, same
seeds — the `--y-ref` mechanism in the driver). Numerics-changing configurations (the nvfp4
combine wire, in-kernel fc2 reduce) are reported separately and labeled. The full table with
every candidate, timing, and bitexact verdict is `results/megamoe_ws8_hillclimb.txt`; the
summary at world_size = 8:

| tokens/rank | best bit-exact config | median | vs baseline | numerics-trading best | median | vs baseline |
|---|---|---|---|---|---|---|
| 128 (decode) | tile (256,**64**,256) + epi_flag (1,2) | 0.3369 ms | **−1.8%** | — (nvfp4 combine is slower here) | — | — |
| 2048 (prefill) | (baseline knobs already optimal) | 0.5992 ms | — | combine=nvfp4 | 0.5295 ms | **−11.6%** |
| 4096 (prefill) | token_back=standalone | 0.9298 ms | −0.8% | combine=nvfp4 | 0.8161 ms | **−12.9%** |

The decode win came straight from the IKET diagnosis: with ~16 tokens per expert, the default
(256,**128**,256) tile pads the token dimension almost 8x; narrowing the token tile to 64 trims
epilogue and combine work without touching weight traffic, and the output stays bit-identical
(`max|d|=0` on every rank). Twelve other knob axes were flat or negative — the GB200-derived
heuristic is essentially already optimal on B300 at decode, which is itself worth knowing.

**Head-to-head against DeepGEMM megamoe** (DeepSeek's own kernel, `deep_gemm` 2.6.1,
fp8-activation x fp4-weight wire — a different quantization wire, so NOT bit-comparable with
the CuTeDSL kernel; both are real DSV4 serving configurations; both kernels run their own
upstream heuristics; same CUPTI benchmark, same machine, same geometry):

```
tokens/rank    cutedsl best         deepgemm     verdict
   128         0.3369 ms (bitexact) 0.3217 ms    deepgemm +4.5% faster
  2048         0.5295 ms (nvfp4)    0.7506 ms    cutedsl 29.5% faster
  4096         0.8161 ms (nvfp4)    1.3620 ms    cutedsl 40.1% faster
```

CuTeDSL + nvfp4 combine wins prefill decisively on B300 — including the 2048-token case that
the sglang PR #31470 sweep had CuTeDSL losing on GB200 before autotune. DeepGEMM keeps a small
decode edge; IKET says decode is a weight-bandwidth floor and the knob space is exhausted, so
closing that last 4.5% needs kernel work (DeepGEMM's ring-coupled FC2→combine is the plausible
mechanism, per its P99 behavior in sglang #31470), not tuning.

**A fairness note on the benchmark closure.** Our first CuTeDSL closure included a
device-to-device copy of the output into a caller-owned tensor, while DeepGEMM's kernel writes
its output tensor directly. The shim already ships the fix — `nvfp4_mega_launch_thunk`, a
prebuilt bare-kernel launcher with no per-call Python and no output copy (the same
workspace-view semantics as open flashinfer PR #4341, which measured +2.6–6% decode-dominated
output throughput in sglang #33571). The final numbers below use that copy-free closure for
CuTeDSL, verified bit-exact against the same reference.

<!--MEGAMOE_VIEW_SHOWDOWN-->

**IKET before/after of the winning prefill config** (m=2048, bf16 vs nvfp4 combine wire):

<!--MEGAMOE_IKET_BEFORE_AFTER-->

## Step 10: what NCU can see here, with sudo — and what it cannot

This machine restricts GPU performance counters to admin (`RmProfilingAdminOnly: 1`), so ncu
(2026.1.0) ran as root with extended privileges inside the container. The result is the
strongest argument for in-kernel tracing we found all night:

1. **A multi-pass NCU collection deadlocks on this kernel class.** NCU's kernel-replay model
   re-executes a kernel to collect more counters than fit in one pass. The megamoe kernel
   contains cross-rank NVLink barriers: a replayed kernel waits for peer ranks that are not
   re-running, and the profile hangs forever. Raw log (`logs/megamoe_ncu_m128.log`) — eight
   `==PROF== Connected` lines, GPUs allocated, zero progress until we killed it.
2. **Even a single-metric, single-pass collection stalled.** We retried with nothing but
   `--metrics gpu__time_duration.sum` (`--launch-count 1`) under a 600-second guard. NCU
   serializes the profiled launch while it arms collection, the peer ranks run ahead and then
   spin inside their own kernels waiting for the profiled rank at the NVLink barrier, and no
   rank ever completes. The run ended with `==PROF== Trying to shutdown target application`
   when the guard fired (`logs/megamoe_ncu_minimal.log`). On this kernel class, with sudo and
   root and privileged containers, the best NCU could deliver was: nothing.

Even in the best case, NCU's unit of observation is the whole kernel: one row of aggregate
counters for a launch that internally contains dispatch, two grouped GEMMs, an activation, and
a cross-rank combine. IKET's unit of observation is a named phase on a warp. For megakernels,
that difference is not a convenience — it is the difference between a profiler that works and
one that structurally cannot.

## Step 11: how to look at the traces yourself

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
7. **The 4.7.0 wheel does not ship `--enabled-cluster`.** The documentation describes it, but
   `run-iket profile --help` in nvidia-cutlass-dsl 4.7.0 has no such flag — full-grid dumps
   are the only option, and postprocessing time scales with total warp-event count (our
   m=8192 AllReduce cells took the postprocessor hours; plan trace sizes accordingly).
8. **Megamoe knobs plumb through the workspace, not the kernel config.** Passing `knobs=` to
   `Nvfp4CutedslMegaMoeConfig` and launching through the shim silently does nothing; the shim
   reads knobs from `get_symm_buffer_for_mega_moe(knobs=...)`. Our first sweep produced
   bit-identical medians for every "candidate" before we caught this.
9. **NCU cannot kernel-replay a cross-rank-coupled kernel** (see Step 10) — a replayed kernel
   spins on peer barriers forever. Single-pass metric sets or application replay are the only
   options, and application replay does not compose with torchrun rendezvous.

## Repository layout

- `patches/iket-instrumentation.diff` — every kernel edit, applies onto flashinfer `b8c21928`
- `scripts/` — four drivers plus two analysis scripts
- `logs/` — full raw stdout/stderr of every profiled run
- `traces/` — Perfetto traces (`.pftrace.gz`) and JSON traces (`.trace.json.gz`) per experiment
- `results/` — raw analyzer outputs quoted in this README
- `COMM_KERNEL_SURVEY.md` — all communication kernels in flashinfer, classified by IKET-ability

IKET is experimental; its API and output format may change. See the CUTLASS documentation page
"IKET Profiling (In-Kernel Event Tracing)".
