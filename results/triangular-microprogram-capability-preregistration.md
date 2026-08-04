# Triangular microprogram layer — capability preregistration

Status: **withdrawn after independent review; no learnability run authorized**  
Date: 2026-07-27

## Architectural claim

A triangular microprogram layer (TML) retrieves `L` input/output singleton
features from a shared bank and one compact program:

\[
a_i=u_{e_i}^\top x+b_{e_i},\quad
c_i=s_{e_i}\phi\left(a_i+\sum_{j<i}T^{(p)}_{ij}c_j\right),\quad
y=x+\sum_i c_i v_{e_i}.
\]

`T=0` is exactly an additive PEER-style layer.  Setting
`T_ij = u_i^T v_j` is exactly the compiled ordered rank-one residual program.
Learning `T` independently makes the stored program a tiny nonlinear circuit
over reusable input/output features.

This changes where fixed bytes are invested.  Additive PEER spends nearly all
of them on more independent `D`-vectors.  TML spends some on many small
context-specific feature-composition graphs.  The hypothesis is that language
reuses features more often than it needs wholly independent vectors.

## Frozen resource comparison

At `D=4096`, BF16, `L=8`, `N=16,384`, and `P=1,000,000`:

- shared `u/v/bias/scale` bank: 268,500,992 bytes;
- each program: eight `uint16` IDs plus 28 BF16 couplings = 72 bytes;
- program table: 72,000,000 bytes;
- total before router/alignment: 340,500,992 bytes;
- active expert payload: 131,104 bytes/token;
- vector work: 65,536 MACs/token;
- triangular work: 28 scalar MACs/token, 0.0427% logical overhead.

The strict equal-byte additive control therefore receives 20,777 singleton
experts instead of 16,384.  The candidate is not compared to a smaller PEER
bank and does not call combinatorial route count free information.

## Frozen learnability worlds

Use identical examples, seeds, optimizer-step budget, batch schedule, activation,
and router-visible information.

1. **Ordered composition:** apply a sequence of reusable latent operations;
   evaluate seen lengths, unseen permutations, and longer compositions
   separately.
2. **Additive bag:** the target is invariant to operation order and is generated
   by an additive singleton bank.
3. **Dense exception:** targets come from a random matched dense nonlinear map,
   testing whether the structural prior merely wins its own generator.
4. **Mixed world:** ordered and additive examples are interleaved and identified
   only through the same input available to every model.

Models:

- TML with learned triangular couplings;
- TML with `T=0` (additive endpoint);
- equal-resident-byte additive PEER with its larger expert bank;
- direct sequential rank-one programs as a numerical/optimization control;
- matched dense residual MLP at the nearest non-exceeding active MAC and byte
  budget.

Run at least three seeds.  Report train and held-out metrics per world; aggregate
metrics may not hide a failed slice.

## Promotion gates

1. TML improves held-out ordered-composition error by at least 10% relative to
   the stronger of equal-byte PEER and the matched dense control in every seed.
2. TML is within one pooled baseline standard error of the stronger control on
   additive-bag and dense-exception slices.
3. TML beats both controls on the mixed-world median and on longer held-out
   compositions; no seed may regress by more than 2% relative.
4. Compiled TML and direct sequential outputs agree at the derived-coupling
   initialization within the dtype thresholds already frozen for H100 G1.
5. Router, IDs, couplings, gathered vectors, optimizer state, and all training
   compute are reported.  Training cost may be higher, but serving state and
   active work may not be omitted.

Passing this synthetic screen admits a small fixed-data language-model A/B.  It
does not itself establish general language capability.

## Withdrawal reason

This document is not executable as a frozen capability protocol.  Independent
review found that it does not specify generators, splits, dimensions, losses,
steps, or router equality sufficiently; a per-program learned `T` cannot claim
unseen-permutation or longer-program generalization; and the square-activation
degree witness does not separate a deployed SiLU TML from routed PEER.  Any
replacement must use identical routing and program bytes for recurrent and
parallel controls, include preactivation and postactivation triangular controls,
and separate stored-program lookup from generated composition.  No positive
capability claim may be made from the G0 algebra JSON.

## Collision boundary

PEER supplies shared singleton experts but aggregates them additively.
UltraMemV2 uses one-neuron FFN value processing but the inspected formulation
does not attach a retrieved lower-triangular nonlinear circuit to a reused
feature tuple.  Stored-program neural memories and Neurocoder establish the
broad idea of retrieving neural programs.  The retained sliver is the specific
PEER-compatible scalar triangular circuit, its exact endpoints, and its
two-vector-pass executor.  Classical compact WY formulas cover linear products
of rank-one transforms, not learned arbitrary-activation program couplings.
