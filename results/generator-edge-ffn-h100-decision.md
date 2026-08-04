# Generator-edge FFN — H100 decision

## Verdict

**Reject the fixed graph candidate before language training.  Advance the
preregistered self-product control as a separate candidate.**

Result hashes:

- Stage 0: `311895efd322f0202e15319468b2540cddb4c887b489862cdb4dae87e882e657`
- H100: `b937233869fb5b5069da00dbc2113d0627ced5134881f6a430a3edcdf5822160`

## Stage-0 result

Every arm used 36 raw parameters in the deterministic `D=3,M=4` function
probe and 1,179,648 FFN projection parameters/MACs at `D=384,M=1024`.

| arm | sampled functional rank | nullity |
|---|---:|---:|
| SwiGLU | 32 | 4 |
| width-1,536 self-product | **36** | 0 |
| duplicate edge | 24 | 12 |
| two-permutation generator edge | **36** | 0 |

The candidate therefore did convert SwiGLU's scale-null directions into
identifiable functions.  The graph also had exactly 2,048 unique edges with
in/out degree two.  Four algebra tests and two full-model CUDA tests passed.

## Unfused whole-model H100 result

Median latency, 30 randomized interleaved samples after five warmups:

| batch x tokens | parallel SwiGLU | self-product | duplicate edge | generator edge | candidate ratio |
|---|---:|---:|---:|---:|---:|
| 1 x 1 | 6.733 ms | **6.337 ms** | 7.115 ms | 7.102 ms | 1.0548x |
| 1 x 512 | 7.251 ms | **6.897 ms** | 7.699 ms | 7.709 ms | 1.0632x |
| 8 x 512 | 7.810 ms | **7.700 ms** | 8.218 ms | 8.213 ms | 1.0516x |
| 32 x 512 | 11.245 ms | **11.096 ms** | 13.958 ms | 13.825 ms | 1.2294x |

Generator-edge failed the frozen `<=1.05x` gate in all four cells.  The
failure is structural in the reference executor: fixed gathers and the
materialized `2M` edge tensor cost more than the saved input projection,
especially in the large prefill cell.  A fused online down kernel is not
mathematically impossible, but it would be a new systems branch and is not
authorized as a repair here.

## Surviving control

The self-product control

```
a = A x                    # width 1.5 M
y = B (SiLU(a) * a)
```

has the same `3DM` learned weights and dense MACs as width-`M` SwiGLU, full
sampled functional rank, tensor-core-aligned width 1,536, no gather/index
state, and was 1.3% to 5.9% faster than parallel SwiGLU in every measured
cell.  It is not evidence of better language modeling yet.  It is admitted as
the only successor from this branch.

Masked GLU and recent power activations are adjacent prior art, so no novelty
claim attaches to the algebra.  The next experiment asks only whether trading
independent gate/value factors for 50% more independently read self-gated
atoms improves a strong parallel language model at the same dense budget.
