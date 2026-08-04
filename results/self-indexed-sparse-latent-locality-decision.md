# Self-indexed sparse latent — natural-locality decision

Status: **closed at the first natural-language gate**  
Date: 2026-07-30  
GPU: AWS `g7e.4xlarge`, RTX PRO Server 6000 96 GiB

## Decision

Do not train or build a sparse-graph kernel for this branch.  Fixed-size
address-carrying feature state did not survive contact with natural language
representations.

All nine frozen layer-pair gates failed.  The failure is large rather than a
threshold boundary:

| model | top-64 energy range | graph recall@64 range | lift over global | lift over shuffled |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 58.7%-69.0% | 23.6%-28.4% | 3.9-5.6 pp | 5.9-8.4 pp |
| SmolLM2-360M | 46.4%-63.4% | 14.4%-25.7% | 4.3-7.2 pp | 5.4-9.6 pp |
| SmolLM2-1.7B | 33.5%-46.4% | 17.1%-19.5% | 8.1-9.0 pp | 9.8-10.5 pp |

The frozen requirements were at least 90% energy, 80% recall, and 20
percentage-point lifts over both controls at every pair.  No pair passed any
of those four requirements.

## Scaling finding

The strongest possible rescue would have been a favorable width trend.  The
opposite occurred.  At the 25%, 50%, and 75% depth locations, median top-64
energy changed across 135M, 360M, and 1.7B as follows:

- 58.7% -> 46.4% -> 33.5%;
- 59.1% -> 49.5% -> 33.8%;
- 69.0% -> 63.4% -> 46.4%.

Graph recall also failed the non-decreasing-width gate at all three depths.
At 1.7B, the lower bound on complete MLP energy captured by correctly predicted
events was only 9.3%-16.4%.

## What this rules out

The Stage-0 executor proved that word-addressed learned memory can separate
resident capacity from active work.  This run identifies the missing resource:
natural hidden computation does not arrive as a bounded set of stable local
addresses.  A global selector, a growing active set, a growing degree, or a
new sparsity-enforcing training architecture is required.  Any of those is a
different proposal and must pay its own routing, quality, and hardware costs.

This does not rule out PEER/memory layers, which retain a dense controller and
global retrieval mechanism.  It rules out the attempted next step: removing
that controller by carrying a fixed 64-address state through a degree-16 graph.

## Integrity

- result SHA-256:
  `9686144735193ba37f056edbcb806955831c0705445890611c6b06ca7ecf3318`;
- source SHA-256:
  `7e89a579dd1b5e2ac72a1378a9e3561982b0b20abcac34bc0dbfb6e6d87d1499`;
- preregistration SHA-256:
  `dc45248f142c2de78d14ed54f3f9e486b6207134d6cda0b59686762be26e2891`;
- pinned model and data revisions are recorded in the result and
  preregistration;
- focused tests: 3 passed on the target G7e runtime.

