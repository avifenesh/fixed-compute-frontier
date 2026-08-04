# T32R rank-32 boundary scan — CPU decision

Status: **PASS ALGEBRA; ADMIT STREAMING H100 BLOCK MICROBENCH**  
Date: 2026-07-31

## Verdict

The frozen rank-32 operator is internally valid and fits its parameter and
multiplication envelopes. It advances only to a physical streaming-kernel
comparison. Natural learnability and model training remain untested.

## Evidence

- four unit tests passed;
- all random-world arrays and outputs were finite;
- repeated execution was bit-identical;
- the no-handle path returned the boundary state bit-exactly;
- the explicit `R_A=I,R_B=-I` witness negated under record swap;
- the loop reference and vectorized implementation differed by at most
  `1.49e-8`, below the frozen `1e-6` tolerance;
- controlled 128-position softmax worlds at score gaps 2, 4, 8, 12, and 16
  matched the closed-form target weight and stayed within the theorem error
  bound;
- total allocation was 943,538 entries, leaving 24,142 inside the removed
  967,680-entry dense budget;
- exact scan work was 3,230,144 multiplies, below the 3.7M envelope;
- zero corpus, evaluator, model-weight, or GPU access occurred.

## Physical requirement exposed by the pass

A naive implementation materializes two `128 x 384` embedding tensors per
request. That is not an acceptable proof of unchanged activation/workspace
memory. The H100 implementation must stream fixed-size token chunks and update
the softmax statistics online:

```text
m' = max(m, max(chunk_scores))
l' = exp(m-m')*l + sum(exp(chunk_scores-m'))
y' = exp(m-m')*y + sum(exp(chunk_scores-m')*chunk_values)
summary = y/l
```

One-token overlap supplies the three-tap context at chunk boundaries. The
result must match the full CPU reference within a frozen tolerance without
materializing all projected record tokens.

## Next comparison

Benchmark actual BF16 blocks, not arithmetic proxies:

- baseline: ten width-1,024 SwiGLUs over the frozen prompt shapes;
- candidate: ten width-940 SwiGLUs plus one streamed record scan after block 3;
- batch 1, 8, and 32; prompt lengths 9, 12, and 31;
- warm steady-state CUDA-graph or compiled execution;
- p50/p95 latency, peak workspace, HBM traffic estimate, and exact output
  checks.

Any p95 or peak-workspace regression at a registered shape closes this exact
physical composition. Kernel fusion may be implemented only if specified
before timings; post-result fusion rescue is forbidden.
