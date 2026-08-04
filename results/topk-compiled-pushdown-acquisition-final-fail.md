# CPKV-TOPK-001 locked decision: acquisition and compression fail

## Formal decision: NO-GO

The frozen `CPKV-TOPK-001` experiment is a decisive negative result. The
learned `S32` pushdown arm did not acquire a uniformly better held-out
algorithm than every same-ledger control, and compiling it to the served `K3`
beam did not preserve the required causal histories. This lineage is closed;
no width, optimizer, threshold, seed, or beam rescue is authorized.

## Completed evidence

- 180/180 frozen training runs completed with no process failures: six arms,
  three families, and ten seeds.
- Total measured training CPU time was 463,292.014 seconds (128.692 CPU-hours).
- The largest family/seed cell used 19,916.931 CPU-seconds (5.5325 CPU-hours),
  below the frozen six-hour ceiling. Peak measured RSS was 472,600 KiB, below
  the 32 GiB ceiling.
- 60/60 immutable locked-evaluation shards completed with no process failures.
- All six family/bin cells failed both the teacher-forced and free-continuation
  decisions: `A1/E1`, `A1/E2`, `A2/E1`, `A2/E2`, `D1/E1`, and `D1/E2`.

## Why it failed

The acquisition gate required every seed to beat every control, median NLL
gain of at least 0.05 nat/token, median delayed-accuracy gain of at least five
absolute points, and no early regression above 0.01 nat/token.

Across the six cells, the worst per-seed NLL gain ranged from -0.413 to -2.006
nat/token, and the worst across-control median NLL gain ranged from -0.102 to
-0.445 nat/token. The worst median delayed-accuracy gain was nonpositive in
every cell, ranging from -0.0172 to 0.0. Maximum early-slice regressions ranged
from 0.131 to 1.593 nat/token, far above the 0.01 limit. Some individual median
NLL comparisons were positive, but none satisfied the all-control,
all-seed acquisition contract.

Compression was not merely a small quality loss. In all four required A1/A2
cells, the minimum across-seed fraction of prefixes with `delta <= 0.01` was
0.0, while maximum mean discarded causal-history mass was 0.9880 to 0.9946.
The proven local read bound was respected (`maximum_bound_excess < 0`),
but it was unusable because its small-discarded-mass premise was absent.
Recurrent `S32 || K3` KL also failed in A1/A2, reaching 0.2313 nat/token against
the frozen 0.02 per-cell ceiling. Since positive S32 acquisition was absent,
compression and continuation retention ratios were undefined or failed and
cannot support a served-model claim.

## Mathematical lesson

The theorem established only a conditional fact: if the selected beam retains
all but `delta` causal-history probability mass, the renormalized read error is
at most `2 delta`. Training did not make the path distribution sparse enough
for a constant-width beam. The missing result was therefore not a tighter
error proof or a better compiler; it was an acquisition mechanism that both
outperforms dense/recurrent controls and concentrates useful causal histories
before compilation. Under this fixed ledger, learned finite-beam pushdown
state does neither reliably.

This is not a smarter production LLM and does not advance to kernel work. The
next research direction must return to the algebraic problem rather than tune
this implementation.

## Bound artifacts

- Manifest SHA-256: `c48cf230e86162e598a87d1b3c9eb0b5714066195d044341685ab9ecbd21060e`
- Oracle SHA-256: `c5352de03d6d7800fe4076b895cde130cd8de38b4f24de2eacde8a0406fdac7e`
- Checkpoint-freeze receipt SHA-256: `10749e85bcd6c699aa8e277c9db92126d0e40d681ed1684756eba725a2e27c63`
- Locked-evaluation receipt SHA-256: `44cf941ec4cfb5c34beeb93ad9bf4524a2ec580ee22ab12d26f44483557df944`
