# PEA versus IPA — learned structured screen preregistration, revision 4

Status: re-frozen before formal execution  
Date: 2026-07-27

Revisions 1-3 were never formally executed. Independent pre-run review found
that its distributed task repeated only the queried label, enabling 93.75%
key-blind accuracy, and identified Inclusive-Power Attention (IPA) as a
strictly cheaper mandatory control. A second design review then found a weaker
63.67% majority shortcut, a maskable partial-chunk guard, and a conditional
selection loophole. Revision 3 still gave a strong spike the invertible value
`-y` and did not fatally gate the affected origin-8 head. Revision 4 fixes
these remaining shortcuts before observing any formal learned result.

## Question

Can a small model learn a non-pairwise attention operator that selects a
relevant neighborhood sharply but reads its payload tokens softly? If so, is
PEA's extra within-chunk dispersion statistic useful, or does IPA obtain the
gain from ordinary online-softmax state?

The query identifies an anchor key while a random binary answer is stored only
in a nearby payload. A one-read ordinary head can score the anchor or read
undifferentiated payloads, but cannot let the anchor increase only neighboring
payload weights. Both candidate operators can; global temperature two is the
pairwise matched control.

Frozen prerequisites:

- `results/staggered-power-evidence-stage0.json`, SHA-256
  `c9f0874d19f4c520a9f70c2b2eef8adc43475abcf46d6e42f286e43e45ef72aa`;
- `results/inclusive-power-attention-stage0.json`, SHA-256
  `98217c897f358647f81eaf27358c03b4a00e8e114306f8e2b6f20ed2e0935659`.

## Frozen model

- one learned attention read and no contextual encoder before it;
- sequence length 128, input width 21, four heads, Q/K width 16 and value
  width 8 per head;
- independent bias-free Q, K, V, and output projections;
- 16 possible key identities with fixed one-hot/type/value features;
- equal learned scalar counts and byte-identical initialization within seed;
- heads 2-3 are ordinary in every arm;
- candidate heads 0-1 use origins 0 and 8 when staggered;
- no arm receives a learned gate, embedding, normalization, or extra state.

## Frozen arms

1. `ordinary`: four ordinary heads.
2. `temperature2`: heads 0-1 use global score temperature two.
3. `pea_staggered`: heads 0-1 use PEA2 at origins 0 and 8.
4. `ipa_staggered`: heads 0-1 use IPA2 at origins 0 and 8.
5. `pea_same_offset`: both candidate heads use PEA2 at origin 0.
6. `ipa_same_offset`: both candidate heads use IPA2 at origin 0.

PEA chunk router mass is `Z2=sum exp(2s)`. IPA mass is `Z1^2/n`. Both
retain the temperature-one local read. IPA uses only finalized ordinary state
`(m,l,o,n)`; PEA additionally requires `l2`.

## Frozen task mixture

Every example has eight random key-label records and a queried key. Unused
positions are exact zero fillers.

- `anchor_same`: queried anchor and adjacent payload share chunks under both
  origins.
- `anchor_boundary`: anchor and adjacent payload cross exactly one origin, so
  one staggered head retains the relation.
- `exact_copy`: queried key and label occupy the same token. Half of target
  records are in the two origin-8 partial edge chunks and half in full interior
  chunks; this exposes a partial-chunk sawtooth.
- `distributed`: exactly four keys have positive targets and four negative,
  so key-blind aggregate value is exactly zero. Each key has four moderate
  (`strength=0.5`) shares `noise_i + 0.25y` together in its own consensus
  chunk, where four Gaussian noise draws of scale 8 are centered to sum zero.
  Their sum is therefore exactly `y`. Each key also has one strong
  (`strength=1`) spike in the next chunk whose independent sign is drawn from
  a separately balanced four-positive/four-negative assignment. Every key has
  five occurrences and every chunk has four consensus records plus one spike.
  A threshold on one share or the spike is weak while aggregating the four
  moderate shares is exact. This is the learned consensus-versus-spike
  counterpart to the frozen Stage-0 witness.

Train data contain equal task proportions. Validation contains 4,096 fixed
examples per task and is shared across arms within seed. Batch order is shared
across arms. Exact-copy edge/interior subgroup counts and accuracies are
reported.

## Frozen optimization

- seeds: 1103, 2207, 3719, 6673, 9011;
- 65,536 generated training examples per seed;
- batch 512, 800 AdamW steps, learning rate `2e-3`, betas `(0.9,0.95)`, zero
  weight decay, gradient clip 1.0;
- FP32 model and loss on the retained H100;
- terminal evaluation and candidate-only fixed-weight router-power-to-one
  execution ablation;
- raw untransformed score RMS and absolute 99th percentile per head/task;
- audited source, preregistration, and both prerequisites are pinned by
  `results/staggered-power-evidence-learning-integrity.json`; arguments are
  source-enforced and runtime versions are recorded in the result.

Formal mode must use the frozen arguments, the default formal output path, a
device name containing `H100`, and exact integrity-manifest hashes. A
non-formal run is forbidden from writing the formal output.

## Global fatal gates

1. Formal configuration and both prerequisite hashes/pass flags are valid.
2. All arms have identical parameter counts and byte-identical initialization
   within seed.
3. Every reported training, evaluation, subgroup, raw-score, and ablation
   metric is finite.
4. On each seed's fixed distributed validation set, the best empirical
   threshold classifier using the strong spike, a pooled moderate share, the
   maximum moderate share, or the minimum moderate share is at most 56%; the
   sign of the four-share sum is at least 99.9% accurate; and key-blind global
   stored-value sum, accumulated in float64, is within `1e-5` of zero. These
   generator oracles are recorded and must pass before any formal training.

## Candidate fatal gates

Each candidate (`pea_staggered`, `ipa_staggered`) is judged independently:

1. Mean accuracy on each anchor task is at least 10 percentage points above
   ordinary and temperature-two, and the candidate beats both in at least four
   of five paired seeds.
2. On `anchor_boundary`, staggered mean accuracy is at least 5 points above
   its same-offset arm and wins at least four seeds.
3. On `exact_copy` and balanced `distributed`, mean accuracy is no more than 2
   points below ordinary and no individual seed is more than 4 points below.
4. Mean four-task accuracy exceeds ordinary and temperature-two in at least
   four seeds and in the five-seed average.
5. Executing the trained candidate heads as ordinary lowers mean anchor-task
   accuracy by at least 5 points in at least four seeds and on average.
6. For every seed, task, candidate head, and each of raw-score RMS and absolute
   q99, the candidate/control ratio is at most 2. The denominator is the larger
   corresponding ordinary or temperature-two statistic.
7. Before output mixing, record target-token attention separately for every
   head on exact-copy edge and interior examples. For candidate heads 0-1,
   their mean attention's ordinary-relative log edge/interior distortion must
   average within `log(1.03)` and remain within `log(1.08)` every seed. The
   affected origin-8 head is also gated directly: its ordinary-head-1-relative
   log distortion must average within `log(1.05)` and remain within
   `log(1.15)` every seed. All ratio operands must be finite and strictly
   positive; all derived ratios/logs are included in the global finiteness
   gate. This prevents another head, learned output mixing, or classifier
   saturation from hiding partial-chunk bias.

## Frozen selection rule

- A passing IPA is selected by default because it does not require `l2`.
- PEA can be selected only if PEA passes, its mean accuracy averaged over the
  two anchor tasks is at least 5 points above IPA, and its mean accuracy
  averaged over exact-copy and distributed is no more than 1 point below IPA.
  These direct PEA-over-IPA requirements apply even when IPA misses a separate
  absolute gate.
- If neither candidate passes, the branch stops.

Passing selects only a mechanism for a fixed-data tiny language-model screen.
It does not establish equal serving cost, language quality, a production
kernel, free head assignment, or a general model improvement.
