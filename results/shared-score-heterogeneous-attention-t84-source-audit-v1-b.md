# T84 source audit v1 B

Bound source manifest SHA-256:
`efc659106c0d49d9792b547783733440f4dd3a3d4c0c04d60559cb8e8579577a`

Method: static source and manifest inspection only. No import, compilation,
test, fixture generation, timing, model execution, GPU use, or rental.

Verified in v1:

- all source-manifest hashes matched;
- frozen 106-statement family, 16 paired seeds, simultaneous critical value,
  20 percent relative boundary, absolute boundary, and control floor;
- Tropical equations, Sattolo derangement, recursive shuffle, lesions,
  parameter counts, timing schedule, and health calculations.

Blocking findings:

1. Every attention architecture computed both full reducers before dispatch,
   and timing retained diagnostic trace tuples, making the comparison unfair.
2. Receipt checks did not prove actual source equals manifest equals receipt in
   workers.
3. OMP and MKL thread variables were not rejected unless exactly one.
4. Stage 0 used a separate operator instead of the production reducers.
5. A ledger could write CONTINUE and only then discover a resource crossing.
6. Candidate health failure decoded as protocol invalid instead of rejection.
7. Decoder tests covered only six cases rather than all 32 inputs.

Verdict: **REVISE**. This manifest authorizes no execution.
