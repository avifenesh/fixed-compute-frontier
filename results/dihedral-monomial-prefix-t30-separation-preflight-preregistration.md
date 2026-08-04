# T30 arbitrary-state separation — exhaustive preflight preregistration

Status: **FROZEN BEFORE IMPLEMENTATION OR EXECUTION**  
Date: 2026-07-31

## Purpose

Verify the corrected deterministic witness from the T30 pre-run audit.  This
is a pure CPU algebra microbenchmark.  It contains no learned model and makes
no language, novelty, hardware, or production claim.

## Frozen domain

- width `m=109`;
- every one of the `2m=218` dihedral elements `(s,r)`;
- canonical generator words over `R=(+1,1)` and `F=(-1,0)`;
- centered basis inputs `H={+sqrt(m)e_j,-sqrt(m)e_j}`;
- target action `h -> P_g h`;
- relaxed control: each group element receives its own independently optimal
  diagonal-affine map `D_g h+c_g`.

The relaxed control strictly contains every word map produced by a shared
diagonal-affine token recurrence, so its optimum is a valid lower bound for
the named baseline.

## Frozen theorem and expected values

The centered basis has mean zero and covariance identity.  Therefore

```text
NMSE(P,D,c) = (||P-D||_F^2 + ||c||_2^2) / m.
```

The optimum is `c=0`, `D=diag(P)`, and its value is `1-f(P)/m`.

For odd `m=109`:

- identity fixed points: 109;
- each of 108 nonidentity rotations: 0;
- each of 109 reflections: 1;
- uniform relaxed-diagonal lower bound: `108/109`;
- exact routed construction error: zero.

## Mandatory gates

1. all 218 canonical words fold to their declared group code;
2. all 218 group actions are distinct;
3. `R F` and `F R` produce different actions;
4. the centered basis mean is zero and covariance is identity within `1e-12`;
5. brute-force relaxed diagonal NMSE matches `1-f(P)/109` within `1e-12`
   for every group element;
6. the average relaxed diagonal NMSE matches `108/109` within `1e-12`;
7. the constructive routed action matches the target exactly for every group
   and centered-basis input;
8. the state ledger remains 109 BF16 values plus one uint16 frame = 220 bytes.

Any failure rejects the proof implementation.  A pass only confirms the local
operator witness and does not authorize a training or GPU run.
