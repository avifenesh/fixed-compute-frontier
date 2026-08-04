# Independent implementation audit — radial-trust Cayley B1

Date: 2026-08-01  
Reviewer: independent subagent `tesr_claim_review`  
Verdict: **PASS — frozen for one CPU-only execution**

Audited manifest SHA-256:
`795828f71616e622efbd1bde0ac9920d9376a6ae1f8cd15c2362db3d1592483a`

Audited executable SHA-256:
`cd37f67ea5fc947f9d87b9aa49c42ed28c30927c5e468a7d322d22a9dad21d02`

Audited test SHA-256:
`ba053cf2051da1e27de822f2dc046c94b1882f47e3a62043b92d3a80f488a012`

## Review findings closed before execution

- The small Jacobian/JVP/VJP block uses a true FP64 implementation and rejects
  every nonfinite intermediate explicitly.
- Protocol integrity compares every imported file against this independently
  frozen manifest and fails before the census if any hash moves.
- Natural route bits and heap node IDs are reconstructed from independent
  formulas rather than the trace recurrence itself.
- Every cell reports depth-stratified statistics and an exact sorted-radius
  representation of its empirical CDF.
- The vectorized per-token forced-bit lift is semantically equivalent to 512
  applications of the existing tuple mechanism because the block is
  token-separable; the test exhausts every path on a depth-three tree.
- The final full receipt is measured after persistence and never mutated after
  a pass.  Crossing a wall, RSS, or combined-size ceiling replaces it with one
  terminal inconclusive receipt.

This audit authorizes exactly one execution with `CUDA_VISIBLE_DEVICES=''`.
It does not authorize a GPU rental or language training.
