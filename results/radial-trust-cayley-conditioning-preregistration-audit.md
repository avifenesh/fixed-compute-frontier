# Independent audit — radial-trust Cayley B1 conditioning preregistration

Date: 2026-08-01  
Reviewer: independent subagent `tesr_claim_review`  
Audited preregistration SHA-256:
`5519f3bbb9622a28e48acf8f25d9a7bf7b629b1c8706669b213830c7b76a9bc7`

## Verdict

**PASS**

The final audit found no blocking issue.  The approved paper:

- fixes the `D=2` hand witness;
- makes initialization, route hashing, construction order, and independent
  random streams reproducible;
- limits `D=4096` to an honest single-application dtype reference;
- separates invalid/inconclusive harness outcomes from scientific kills;
- keeps raw-radius occupancy as the sole empirical unknown;
- treats forced routes as exact per-input conditional censuses rather than IID
  samples.

This pass authorizes implementation and one CPU-only execution under the
frozen preregistration.  It does not authorize a GPU rental, B2 execution,
language training, or a capability claim.
