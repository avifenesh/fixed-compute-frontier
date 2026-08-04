# Address-factored transport attention: learned-screen preregistration

## Admission and claim

The frozen AFTA Stage 0 passed all ten gates and was independently validated.
This learned screen asks whether a hybrid layer can convert the score-locked
record-retrieval advantage into learned capability without sacrificing the two
cases where independent or ordinary heads are required.

Passing supports only a tiny language-model screen. It does not establish
language-model quality, runtime parity, or novelty.

## Equal core-tensor ledger arms

Every arm has identical learned tensors, initialization, parameter count,
training batches, batch order, optimizer, output width, QK/PV scalar MACs, and
K/V width. Total Q/K/V width is `D=192`; output projection width is also 192.

- `split12`: twelve independent width-16 heads; relation offsets `0,1,2,3`
  repeat three times.
- `split12_temp2`: the same arm with every pre-softmax narrow logit multiplied
  by two.
- `hybrid_afta`: one width-64 Q/K map serving four width-16 V groups with
  offsets `0,1,2,3`, plus eight protected width-16 independent heads whose
  offsets repeat `0,1,2,3` twice.
- `hybrid_common1`: the same hybrid, but every group under the wide map uses
  offset one.
- `hybrid_ordinary0`: the same hybrid, but every group under the wide map uses
  offset zero.

For the hybrid wide map, if four width-16 score blocks are `S_r`, the executed
score is `(sum_r S_r)/sqrt(4)`, exactly equal to a width-64 dot product with
standard scaling. Every transported output uses the original V tensor with
invalid shifted loads zero and no group renormalization.

The candidate is deliberately hybrid. The Stage-0 negative control proved that
one shared address cannot replace independent heads.

This is an equal core-tensor ledger, not an equal runtime claim. A split arm has
12 attention maps per query; a hybrid has nine. Thus a hybrid has 25% fewer
score/softmax elements and, if maps are materialized, 25% less attention-state
storage. Every arm performs twelve full-length width-16 transported reductions,
but the hybrid wide map has four channel-group value regions and AFTA gives them
four distinct shifts. The result records map count, score/softmax elements,
logical nonzero terms, shifted groups, and wide-offset count. Whether those
regions fuse into a serving kernel without a transaction penalty remains an
unproven executor question.

## Data and tasks

Five seeds: `1201, 3253, 4969, 7753, 9901`.

- context length: 83
- records: 16, occupying four payload positions in a five-position cell
- address code: fresh 64-dimensional Rademacher vector per record/example
- query: four independently noised 64-dimensional address views
- payload: every context contains a random permutation of all 16 possible
  four-bit tuples, one tuple per record
- train query noise: `1.25`
- train steps: `1,000`; batch: `256`; AdamW; no dropout
- evaluation examples: `8,192` per task/seed

The address codes are generated afresh and are not a finite learned vocabulary.
The random tuple permutation is independent of addresses and queries. Because
every context contains every tuple exactly once, the target class is uniform
even conditional on the full context when the target address is hidden.

Tasks are uniformly interleaved during training; because 256 is not divisible
by three, per-batch task counts differ by at most one and their order is
shuffled from the frozen data generator:

1. `shared_record`: all four query views name one record; payload bit `r` is at
   relation offset `r`. The target is the ordered four-bit tuple (16 classes).
2. `independent_record`: query view `r` names an independently selected record;
   the target tuple takes bit `r` from that record. This requires independent
   maps and is the protected negative case.
3. `ordinary_record`: all four bits are stored at the addressed anchor token.
   This protects ordinary same-token attention.

Evaluation uses train noise `1.25` plus a shared-record hard-noise evaluation
at `1.75`. The model query contains only the four noisy address views: no
payload bits and no explicit task tag. Task IDs generate batches and metrics but
are never model inputs. In addition to the pre-training data audit, every
trained arm/task is evaluated with all query addresses zeroed and with query
addresses cyclically permuted across examples.

## Model

Each model contains bias-free dense Q/K/V projections from the same input to
192 total channels, the selected attention algebra, a bias-free 192-by-192
output projection, RMS normalization, and a 16-class readout. Only the final
query reads the context. All arms have the same state-dict keys and shapes.

No positional convolution, extra compiler, auxiliary loss, task-specific
classifier, or arm-specific parameter is allowed.

## Diagnostics and ablations

- Hash every arm's initial state and require equality within seed.
- Count parameters and the symbolic QK/PV/cache ledger.
- Audit class balance and query/payload independence from 8,192 generated
  examples per task/seed. The frozen maximum absolute Pearson-correlation bound
  over 256 query-code coordinates by four target payload bits is `0.06`.
- Require both trained-model address-destruction evaluations (zero addresses and
  across-example address permutation) to remain at or below 7.5% accuracy for
  every arm/task.
- Report per-seed accuracy and loss for every arm/task.
- Report raw score RMS and absolute q99 for every attention map.
- Evaluate `hybrid_afta` after zeroing its complete 64-channel wide output
  before the shared output projection.
- Verify every task's inputs and labels are byte-identical across arms for a
  frozen diagnostic batch. Hash the training generator state at every step and
  the complete first and final batches to verify identical training streams.

## Frozen gates

All gates are conjunctive.

1. **Integrity:** Stage-0 hash/pass, formal configuration, finite metrics,
   equal initialization, equal parameters, equal data, and equal ledger pass.
2. **Shared gain:** at noise `1.25`, candidate mean accuracy exceeds the best
   noncandidate arm by at least 5 percentage points and wins in every seed.
3. **Hard shared gain:** at noise `1.75`, candidate mean accuracy exceeds the
   best noncandidate by at least 5 points.
4. **Independent protection:** candidate mean accuracy is no more than 1 point
   below `split12`, and is no more than 2 points below it in any seed.
5. **Ordinary protection:** candidate mean accuracy is no more than 1 point
   below `split12`, and is no more than 2 points below it in any seed.
6. **Overall gain:** candidate mean over the three noise-1.25 tasks exceeds
   `split12` by at least 2 points.
7. **Causal use:** zeroing the candidate's wide output reduces shared-record
   accuracy by at least 5 points in every seed.
8. **Score guard:** candidate wide-map raw-score RMS and absolute q99 are each
   between 0.5x and 2.0x the median `split12` narrow map on every task/seed.
9. **No shortcut:** every audited class prior is at most 7.5%, every
   query/payload maximum absolute correlation is at most `0.06`, and every
   arm/task is at most 7.5% accurate under both zero-address and
   across-example-address-permutation evaluation.

Failure closes this exact `1 wide + 8 protected narrow` allocation. A successor
must be justified by the failure mode; changing head counts or task weights to
rescue the result is forbidden.

## Prior-art boundary

GTA already shares attention maps across heads, attention-map reuse shares maps
across layers, and CAT already uses convolution/value delay. This screen cannot
support claims to map sharing, value delay in general, runtime parity, or
novelty. Its surviving hypothesis is the narrower composition
`y_r = a^T P_r V_r`: one learned content-address map with distinct fixed
relation transports for different value-channel groups.
