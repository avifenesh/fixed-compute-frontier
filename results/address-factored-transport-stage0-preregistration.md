# Address-factored transport attention: Stage-0 preregistration

## Frozen claim

One wide content address can safely serve several narrow payload fields when
the fields share an address but occupy fixed related positions. This can create
more effective relation heads without increasing the total QK or PV matrix
multiplications, projection parameters, output width, or KV-cache width.

This Stage 0 tests only the address-reliability mechanism. It does not claim
language-model quality, kernel parity, novelty, or that records in natural text
always have fixed offsets.

## Operator

Partition a head's `D` value channels into `R` groups. Let

\[
A=\operatorname{softmax}(QK^\top/\sqrt D).
\]

For group `r`, use a fixed prefix-truncated relation shift between key
addresses and value payloads. For query prefix `t` and offset `delta_r`:

\[
Y_{t,r}=\sum_{\substack{0\le j\le t\\0\le j+\delta_r\le t}}
A_{t,j}V_{j+\delta_r,r},\qquad Y=\operatorname{concat}_r Y_r.
\]

Invalid shifted loads are exactly zero and the surviving attention mass is not
renormalized per group. A nonzero global/circular permutation is forbidden: it
would expose a future value on some causal prefix. The truncated shift is a
partial permutation whose valid domain depends only on the current prefix.

We call this address-factored transport attention (AFTA). Ordinary attention
is the contained special case in which every `delta_r=0`. A common value-delay
head is the special case in which every channel uses the same offset.

The equal-matrix-compute control is `R` ordinary transported heads, each with
query/key/value width `g=D/R`. It computes `R` independent maps `A_r` and
returns one field per map. AFTA instead sums all `R` address-evidence blocks
before softmax and reuses the resulting width-`D` map for all fields.

## Exact ledger

For a query/key rectangle of `N_q x N_k`:

| quantity | AFTA | `R` narrow heads |
|---|---:|---:|
| total Q width | `D` | `R g = D` |
| total K width | `D` | `R g = D` |
| total V/output width | `D` | `R g = D` |
| QK scalar MACs | `N_q N_k D` | `R N_q N_k g = N_q N_k D` |
| PV scalar MACs | `N_q N_k D` | `R N_q N_k g = N_q N_k D` |
| K cache scalars/token | `D` | `D` |
| V cache scalars/token | `D` | `D` |
| Q/K/V projection outputs | `3D` | `3D` |
| bias-free Q/K/V/O parameters at input width `M` | `4MD` | `4MRg = 4MD` |
| softmax maps | `1` | `R` |

Both arms additionally carry `R` static offset integers, evaluate one bounds
predicate per query-key relation group, load `N_q N_k D` V scalars, and use `R`
total V load regions of width `g`. These executor costs are recorded separately
from matrix MACs. Equal totals do not imply equal runtime: AFTA places all `R`
regions under one attention map and fragments a wide V tile, whereas the split
control uses one contiguous region under each of `R` maps.

The transport changes only which cached V row supplies each contiguous channel
group. It must address the original cache directly, not materialize a shifted
V tensor, and it does not create or mutate a second KV cache. A production kernel may
still lose throughput from fragmented V loads; that is explicitly deferred to
a later fatal kernel gate.

## Strict functional distinction

For a standard head, all value channels use one address map and the same token
alignment. In AFTA, group `r` uses a prefix-truncated shift of `A`. Thus one selected
address can expose a tuple of features from several graph-related tokens.

Against `R` narrow transported heads, the distinction is evidence fusion. The
wide AFTA logit is

If the standardized narrow logits are
`S_r=Q_r K_r^T/sqrt(g)`, the exact wide logit map is

\[
S_{\mathrm{wide}}=\frac{1}{\sqrt R}\sum_{r=1}^{R}S_r.
\]

whereas the narrow heads normalize each partial score independently. Once the
partial softmaxes have selected inconsistent records, the output projection
cannot reconstruct fields that were never retrieved. AFTA can combine weak
independent address evidence before selection.

The converse matters: if the requested fields have different addresses, one
shared map is the wrong factorization and the narrow heads should win. The
formal screen includes this as a required negative control; any successor must
retain ordinary/independent heads.

## Frozen experiment

Five seeds: `11, 23, 37, 53, 71`.

- records per example: `128`
- relation fields: `R=4`
- address dimensions per evidence block: `g=16`
- total address and payload width: `D=64`
- frozen model/input width for the parameter ledger: `M=256`
- projection biases: disabled
- Stage-0 relation offsets: `0, 1, 2, 3`
- trials per seed and condition: `8192`
- Gaussian query-noise scales: `0.75, 1.0, 1.5`
- address codes: independent unnormalized Rademacher vectors
- payload codes: independent normalized Gaussian vectors of width `g`

K and V have the same 128 sequence positions. For an offset `delta`, only
address rows satisfying `0 <= j+delta < 128` participate; invalid tail mass is
zeroed without renormalization. Targets are sampled only from the valid domain
of their required relation (the intersection of all four domains in the
shared-address condition). No padding V tokens or extra cache rows are added.

### Shared-address condition

All four query blocks identify the same randomly chosen record. Payload field
`r` is stored at the fixed relation offset assigned to group `r`.

- AFTA forms the exact scaled width-64 logit, applies softmax, and uses the one
  resulting attention map to transport all four payload-code groups.
- The split control applies four separately scaled softmaxes to the
  16-dimensional partial logits and transports one payload group per head.
- A wide-ordinary control uses offset zero for every group. A common-delay CAT
  control uses offset one for every group. Both reuse AFTA's same wide map.
- Each returned field is decoded by nearest payload code. Exact-record accuracy
  requires all four decoded fields to come from the target record. Field
  accuracy is also reported.

### Independent-address negative control

Each query block identifies an independently chosen record. The split heads can
retrieve one field from each address. AFTA must use one common selected record
for all fields. This condition is not an intended use case; it verifies the
operator's predicted limitation rather than allowing a universally favorable
benchmark.

### Audits

- The concatenated width-64 dot product must equal the sum of the four partial
  width-16 dot products within the frozen numerical tolerance.
- Every target index and code tensor is generated before either arm is scored.
- Resource formulas enumerate Q, K, V, and O matrix shapes and exact parameter
  counts at the frozen model width and are evaluated by an independent ledger
  function.
- A direct causal operator audit perturbs every value strictly after each
  prefix and requires the current-prefix output to remain unchanged.
- Setting every offset to zero must equal ordinary causal `A @ V` exactly
  within `2e-6` in FP32.
- Forward output and Q/K/V-gradient results must match an independently
  constructed dense effective-map reference within `2e-6`.
- The shared-record witness also reports a common-delay control, which applies
  one offset to every channel group and therefore cannot return the ordered
  four-field tuple.
- Report both exact-record and per-field accuracy; no best-of-seed selection.

## Frozen gates

All gates are conjunctive.

1. **Algebra:** maximum absolute difference between the concatenated raw score
   and summed partial raw scores is at most `2e-10` in the independent FP64
   audit; the executed logits use the frozen FP32 scaling above.
2. **Ledger:** QK MACs, PV MACs, projection output width, parameter-shape
   ledger, and K/V cache scalars are exactly equal between arms.
3. **Shared-address reliability:** at noise `1.0`, AFTA exact-record accuracy is
   at least `90%` in every seed.
4. **Matched advantage:** at noise `1.0`, AFTA exceeds split-head exact-record
   accuracy by at least `25` percentage points in every seed.
5. **Noise robustness:** averaged over all seeds at noise `1.5`, AFTA exceeds
   split-head exact-record accuracy by at least `20` points.
6. **Predicted limitation:** at noise `1.0` in the independent-address control,
   split-head per-field accuracy exceeds AFTA per-field accuracy by at least
   `20` points in every seed.
7. **Causal operator:** future-value perturbation changes no prefix output by
   more than `2e-6`; all-zero offsets equal ordinary causal attention within
   `2e-6`; invalid transport mass is not renormalized.
8. **Common-delay distinction:** at noise `1.0`, AFTA exact-record accuracy
   exceeds the common-delay tuple accuracy by at least `50` points in every
   seed.
9. **Wide-ordinary distinction:** at noise `1.0`, AFTA exact-record accuracy
   exceeds the wide-ordinary tuple accuracy by at least `50` points in every
   seed.
10. **No numerical failures:** every reported scalar is finite.

Passing Stage 0 only admits a matched learned structured-task screen. It does
not admit a language-model or kernel experiment.

## Ancestry boundary

This is not presented as isolated prior-art-free invention. Convolution-
Augmented Transformer already establishes the value-delay/successor-read
primitive with a common convolutional filter. Channel-wise Sample Permutation
already demonstrates different fixed sample permutations across value
channels while replacing content attention. AFTA's research question is their
composition under an exact Transformer ledger: retain one learned, wide
content-address map and apply different fixed relation transports to its narrow
value groups.
