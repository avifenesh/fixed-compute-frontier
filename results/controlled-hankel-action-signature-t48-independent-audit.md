# T48 independent audit — closure confirmed

Date: 2026-08-01  
Verdict: **GO TO CLOSE OPERATOR-SIGNATURE LANE; ZERO RUNS**

The audit confirmed the Hoeffding union bound, Frobenius-to-operator bound,
Weyl singular-value margin, and scoped Bernoulli minimax lower bound. It
required four corrections now applied to T48:

1. quotient joint observation/reward label permutations by sorting the multiset
   of slice singular-value vectors;
2. include rewards/costs in controlled outputs rather than signature only the
   observation process;
3. charge rank-revealing core selection, history/test correspondence, prototype
   estimation, and legal reachability; and
4. make PSR update/planning conditional on a sufficient rank core while keeping
   unconditional containment of the proposed signature by the full Hankel
   slices.

After correction, the signature remains a deterministic lossy function of
standard controlled Hankel data. A full-slice PSR/OOM/system-identification
control has at least the same information and no greater necessary acquisition
cost. Another invariant of the same matrix cannot reopen the lane.
