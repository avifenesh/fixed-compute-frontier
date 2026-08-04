# Address-factored transport: weight-RMS successor preregistration

## Admission and scope

The frozen learned AFTA screen is a valid failed result. It showed a bimodal
wide branch: causally useful in three seeds and unused in two, with wide/narrow
score RMS ratios spanning `0.399x` to `2.157x`. This successor asks one question:
can a training-only per-map Q/K radial constraint make the learned AFTA advantage
reliable while exporting the exact ordinary served graph?

Passing admits only a tiny language-model screen. It does not establish
language-model quality, runtime parity, or novelty. Failure closes AFTA; there
will be no SVD rescue, longer structured run, head-count change, task reweight,
or relaxed gate.

Frozen prerequisites:

- prior learned source SHA-256:
  `3c969819e558007090bc5dfb031bf59d9c268ac75ee57425215bb3fae12cb8a0`;
- prior learned preregistration SHA-256:
  `fb931993d65e34f4e0e493d73c4df8939683c750f16d9d5bbb7cb7f49bcf2739`;
- prior failed-result SHA-256:
  `77a3db085a0187133204543a7682807cb86e04322f2badd3c096f30a75874885`.

## Frozen arms, data, and compute

Reuse the five arms, all three tasks, model dimensions, 16-record complete-tuple
data generator, noise levels, 1,000 steps, batch 256, 8,192-example evaluations,
controls, address-destruction audits, output ablation, and core served ledger
from the failed learned screen.

Use fresh seeds `4144, 2555, 1506, 1060, 2778`. They are the first five
32-bit chunks of the frozen failed-result SHA, mapped as
`1000 + (chunk mod 8000)`; they were not selected from exploratory runs.

No activation QK normalization is allowed. No served operation changes.

## Training-only weight-RMS reparameterization

Raw Q/K weights retain the exact ordinary shapes `[192, 264]`, state keys,
initial values, and learned-parameter count. Map blocks are the blocks actually
executed by each arm:

- split arms: twelve contiguous width-16 blocks;
- hybrid arms: one width-64 block followed by eight width-16 blocks.

Immediately after loading the common per-seed initialization, freeze for each
used block and projection

`rho_0 = sqrt(mean(W_0^2) + 1e-12)`.

Each training forward uses

`W_eff = W_raw * rho_0 / sqrt(mean(W_raw^2) + 1e-12)`

within each executed map block. Autograd passes through the denominator. The 12
Q and 12 K reference slots are nonlearned, nonpersistent training buffers;
unused hybrid slots are zero. Raw Q/K tensors use AdamW with weight decay zero
because radial decay is functionally removed. Every other parameter retains the
frozen `1e-3` weight decay and all optimizer hyperparameters remain unchanged.
No post-step projection or optimizer-state rewrite is allowed.

Before any formal evaluation, materialize every `W_eff` into ordinary dense
Q/K weights, discard the reparameterization buffers, load the exported state
into the original ordinary classifier class, and evaluate only that class.

Training processes and scales the same `2 * 192 * 264 = 101,376` Q/K weight
scalars in every arm. Split arms execute 24 block-RMS reductions per forward
(12 Q plus 12 K); hybrid arms execute 18 (nine Q plus nine K). Every arm
allocates 24 scalar reference slots; split uses all 24 and hybrid uses 18. This
training-only difference follows the executed-map topology and is recorded per
arm in the result. Exported learned parameters, state bytes, QK/PV MACs, K/V
cache, attention maps, score/softmax work, transport topology, and per-token
serving operations are exactly the prior ordinary ledger.

This intervention does not quotient the raw Q/K parameter gauge and does not
establish that radial drift solely caused the failed screen. A pass supports
only that constraining each executed map's effective Q/K Frobenius RMS is
sufficient for this screen under the frozen recipe.

## Frozen gates

The prior nine conjunctive gates are unchanged:

1. integrity;
2. shared gain of at least 5 points over the best noncandidate, with a win in
   every seed;
3. hard shared gain of at least 5 points;
4. independent protection within 1 point on mean and 2 points in every seed;
5. ordinary protection within 1 point on mean and 2 points in every seed;
6. overall gain of at least 2 points over `split12`;
7. at least a 5-point wide-output ablation drop in every seed;
8. wide/narrow raw-score RMS and absolute-q99 ratios within `[0.5, 2.0]` on
   every task/seed;
9. all class-prior, correlation, zero-address, and permuted-address shortcut
   bounds.

Add one conjunctive **export-integrity** gate:

- all 25 seed/arm exports are present;
- maximum pre-export reparameterized versus exported-ordinary logit error is
  at most `2e-6` on the frozen diagnostic batch;
- exported state loads strictly into the original ordinary classifier;
- training and exported learned-parameter counts equal the prior count for
  every arm;
- Q/K raw parameter groups use zero weight decay and all other groups use
  `1e-3`.

The failed screen's extra descriptive phase thresholds are not added as gates;
the unchanged per-seed shared, protection, causal-use, and score guards already
test whether the bifurcation disappeared where it matters.
