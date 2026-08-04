# T38R transport retry — viewer-provenance failure

Date: 2026-08-01  
Decision: **INCONCLUSIVE; NO CORPUS OR HELD-OUT STATISTIC MATERIALIZED**

The independently reviewed T38R retry preserved every scientific constant and
rate-limited the frozen viewer requests.  It opened its separate seal, then the
code stratum returned `partial=true`.  The frozen protocol requires
`partial=false` for every block, so the run stopped before `materialize_corpus`
wrote any file.

Preserved T38R opening seal SHA-256:
`471b65edb64fb82849bd41fe1a189bba51e16c163555c93895ca122973b3da2a`.

Absent after failure:

- T38R corpus;
- T38R manifest;
- T38R finalization seal;
- T38R census result;
- every held-out count, purity, coverage, or advantage statistic.

No third remote retry is admitted.  T38 remains algebraically valid but
empirically unresolved.  A separate local optimistic upper-bound audit may
still close it without pretending to reproduce the sealed census: if even a
copy-rich local corpus has less than the frozen 3% representable `COPY`
coverage, the full candidate cannot pass its own necessary gate.
