# Resource Measurement Contract v0

Each result is scoped to one tuple:

`(model build, server build, GPU stratum, workload cell)`

H100 SXM, H100 PCIe/NVL, H200 SXM, and H200 NVL are separate strata. Results
are never pooled across them.

## Frozen manifest

Before candidate measurement, record:

- exact GPU SKU/PCI ID, count, topology, MIG, CPU/NUMA, power limit, clocks,
  ECC, RAM, NIC, and temperature state;
- driver, firmware, CUDA, cuDNN, NCCL, compiler, engine commit, kernel versions,
  container digest, and relevant environment;
- checkpoint/tokenizer hashes, packed format, cache/state dtype, helpers,
  parallelism, and sharding;
- scheduler, batching, CUDA graphs, allocator/workspaces, prefix cache, and
  admission policy;
- prompt/trace hash, exact lengths, decoding, arrivals, concurrency, prefix
  sharing, cache state, and SLO;
- cold/warm state, compilation/autotuning caches, seed, randomized run order,
  background processes, temperature, clocks, and throttle reasons.

Baseline and candidate run in randomized interleaved blocks on the same physical
GPU set.

## Rental admission

Target hardware is used only for an uncertainty that cannot be resolved by a
reduced local measurement.  Before renting, attach the passing local CPU/GPU
receipt, the same-codepath check, measured scaling estimate, frozen maximum
wall time and dollar cost, and early-stop rule to the target manifest.  If a
small falsification fits locally but the local GPU is busy, wait rather than
renting it elsewhere.

Every rental begins with a bounded smoke block.  Continue into a multi-hour
run only when the smoke result agrees with the local direction and its time,
memory, and numerical behavior lie inside the preregistered envelope.  A
disagreement stops the run and returns the candidate to local diagnosis.

## Exact byte ledger

Report integer bytes for:

- serialized unique learned payload;
- unique learned HBM residency, replica sum, and maximum per GPU;
- host/remote learned bytes and streamed bytes per request/token;
- static runtime memory, request state, workspaces, allocator-reserved memory,
  and peak transient memory.

Scales, zero points, indices, routers, codebooks, adapters, draft/verifier
models, and learned constants count. Aliases are deduplicated by physical
storage identity. Framework allocator snapshots must reconcile with CUDA free
memory; unexplained residuals fail exact accounting.

## Operation ledger

Publish both:

- **F_math:** operations required by the executed dynamic graph;
- **F_issued:** padded/sparse operations represented by issued hardware
  instructions.

Partition by datatype. Count FMA as two FLOPs; list transcendental/SFU, integer,
comparison, selection, top-k, atomics, addressing, and communication separately.
Use actual batches, contexts, routes, padding, rejected drafts, and
recomputation. Normalize prefill by accepted input tokens and decode by accepted
output tokens. Kernels covering at least 99% of GPU time need analytic or
instruction-derived accounting.

## Traffic and persistent state

In separate profiler passes, measure HBM/L2 reads and writes, NVLink, PCIe,
host/network, and explicit copies. Store raw counter names, units, tool versions,
and conversions. Profiling replay is not application traffic and profiling runs
are not latency runs.

Count every byte readable by future decode steps: KV, recurrent state,
summaries, page tables, scales, routing history, prefix caches, speculative
state, and offloaded state. Report exact logical, allocated, reserved, and
transient bytes for every context/concurrency cell, including paging staircases.

## Latency, throughput, energy, and cost

Report separately:

- server-admission and client-visible TTFT;
- TPOT p50/p90/p95/p99;
- accepted input/output throughput;
- maximum offered load satisfying the frozen SLO;
- request failures, queue depth, realized batch distribution, and rejection;
- cold start, compilation, warm steady state, cache hit, and cache miss;
- gross/idle-subtracted joules, watts, clocks, temperature, and throttling;
- gross GPU cost and total system cost per million accepted output tokens and
  per completed request.

Use both deterministic forced-length decoding and natural stopping. Count every
allocated GPU for the full wall interval.

## Instrumentation passes

1. **Timing:** client/application clocks and NVTX with minimal telemetry.
2. **Timeline:** CUPTI or Nsight Systems for kernels, copies, synchronization,
   collectives, and the critical path.
3. **Counters:** CUPTI Range/SASS or Nsight Compute for issued work and HBM/L2.
4. **Telemetry:** NVML/DCGM for energy, clocks, temperature, throttling, and
   utilization.
5. **Memory:** allocator snapshots, CUDA memory queries, and process RSS.

Instrumentation changing throughput or p99 by more than 0.5% is excluded from
timing or charged equally to both systems.

## Repetition and decisions

- At least five fresh-process blocks per system.
- Loaded p99 cells target at least 10,000 requests.
- Continue until 95% interval half-width is <=1% for throughput/median latency
  and <=3% for p99/energy, or label the result inconclusive.
- Block-bootstrap time windows; use simultaneous one-sided 95% intervals.
- Exact bytes, mathematical FLOPs, and logical state have zero tolerance.
- Default maximum noninferiority margins: 2% latency/throughput/cost and 3%
  energy; capability margins come from baseline repeatability and practical
  floors fixed in advance.

Decision labels: **Pareto pass**, **noninferior only**, **conditional pass**
(named hardware/workload cells), **frontier trade**, **fail**, or
**inconclusive**. A Pareto pass requires every protected resource and capability
constraint to pass and at least one confidence interval to clear a predeclared
superiority margin.
