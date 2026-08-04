# T70 local falsifier preregistration — bounded persistent lifetime learning

Date frozen: 2026-08-02  
Status: **SUPERSEDED BY T71 AFTER THE COMPLETED LOOKUP OPERATOR SMOKE; RETAINED AS HISTORICAL CONTROL INVENTORY; NO FURTHER RUN**

The lookup operator smoke permitted by this document has been completed. T71's
objective reset now supersedes the remaining cross-family admission. The
controls below are retained for reuse, but this is not a currently frozen or
executable decisive manifest.

## 1. Claim tested

A fixed-weight, bounded-state learner trained with hard transcript erasure and
cross-boundary lifetime credit can acquire, retain, use, and revise compact task
information across a shared typed meta-distribution. It must transfer its state
update to a family excluded from updater training and beat the strongest
bounded evidence-management alternative at a materially better
capability/resource point.

The claimed improvement belongs to the stateful system. It is not zero-shot
backbone improvement and it vanishes when learner state is reset.

## 2. Concrete intervention

Candidate `BPML-state-token`:

1. a small causal Transformer processes fixed-length typed interaction
   segments;
2. each causal segment is `[s memory-read][event/scratch][s memory-write]`;
3. suffix write tokens attend the complete segment, and only their projected
   outputs persist as the next segment's read tokens;
4. all transcript tokens, scratch activations, and ordinary KV state are erased;
5. loss after the boundary credits earlier state writes; and
6. weights are frozen during evaluation—only the forward state update learns.

Primary causal ablation `no-lifetime-credit` uses the identical model, state
shape, data, optimizer opportunity, and targets. It preserves forward state
across segments but stops its gradient at every boundary and provides no
future-loss credit to earlier writes. A separate `reset-state` unit ablation
sets state to its initial constant and tests the information-channel theorem.

Alternative GRU/SSM writers trained with the same lifetime contract are T70
implementations. The best one selected entirely on development seeds becomes
the frozen candidate before decisive seeds.

## 3. Shared interaction interface and families

Every environment exposes only:

```text
OBSERVE <serialized payload>
PROBE <serialized input> -> FEEDBACK <serialized output>
PREDICT <serialized input> -> scored output
```

There is no family name, environment ID, change flag, task-boundary label,
family-specific prompt, adapter, verifier, or tool. Boundary events are either
visible to every system or invisible to every system. Symbol alphabets,
serialization tokens, order, and padding are freshly permuted after training.

Four structurally different finite families are frozen:

1. **Lookup:** a random finite key-to-symbol map; causal-path/capacity unit.
2. **Affine:** `y=ax+b mod p` for held-out primes and two defining examples;
   systematic sufficient-code unit.
3. **Parity:** an unknown binary linear functional over renamed bit vectors;
   compositional linear-rule family.
4. **DFA:** membership in a hidden small binary automaton queried by strings;
   active experiment-selection and hypothesis-integration family.

The exact domain sizes are selected once on development seeds so the
full-context oracle clears its competence floor, then frozen. Decisive sizes,
generators, and seeds are written to the run manifest before training decisive
models. No size or family may be removed after seeing decisive results.

## 4. Rotated held-out protocol

For each family `j`:

1. establish full-context competence on all four families using the same
   exact per-rotation checkpoint later used by the candidate; full-context mode
   changes only transcript access, not weights or adapters;
2. train the bounded updater on `F\{j}` only;
3. freeze backbone, updater, memory interface, hyperparameters, and thresholds;
4. evaluate on `j` with new rules, symbol permutations, orders, change points,
   and horizons at least `2x` the maximum training horizon; and
5. repeat until every family has served as `j`.

Holding out only instances from a seen family does not count. Also evaluate
factorial held-out combinations of familiar primitives, fixed before decisive
seeds.

## 5. Controls

| control | role | information/state privilege | accounting |
|---|---|---|---|
| reset/no-state | chance and zero-learning floor | current input only | same backbone forward |
| constant state | channel ablation | fixed candidate-sized tensor | same forward |
| independent-environment state | causal shuffle | candidate state from independent hidden rule | same forward |
| no-lifetime-credit | primary causal ablation | identical architecture and bytes | identical training budget |
| raw FIFO examples | episodic control | candidate-matched bytes | reader work charged |
| reservoir examples | long-stream episodic control | candidate-matched bytes | sampling/read charged |
| learned episodic compression/retrieval | strongest memory control | candidate-matched bytes and training data | index/read/write/tuning charged |
| task sufficient statistic | family-specific ceiling | exact lookup table, affine coefficients, parity vector, or DFA learner | labeled oracle, not cost-matched winner |
| full context | information ceiling | complete transcript | context/KV/FLOPs charged |
| TTT/adapter | online-gradient control | equal observations and state-plus-optimizer budget where feasible | backward/checkpoint cost charged |
| Replay-MAML/PTW-style | change-point control | equal feedback/boundary data | replay and `O(log T)` growth charged |

All learned controls receive the same training examples, action/probe budget,
feedback timing, boundary information, parameter-search trials, development
seeds, and decisive seeds. They may allocate their bytes optimally; they are not
forced into latent slots.

## 6. Metrics and gates

For each held-out-family rotation separately:

1. **Oracle competence:** full-context accuracy must be at least `90%` on the
   private query set. Otherwise that rotation is invalid before candidate
   interpretation and no scale follows.
2. **Oracle-gain recovery:** with reset accuracy `A_0`, full-context accuracy
   `A_F`, and candidate accuracy `A_C`, require
   `G=(A_C-A_0)/(A_F-A_0) >= 0.80`.
3. **Lifetime regret:** candidate cumulative scored loss plus declared probe
   cost must be at least `30%` lower than the strongest non-oracle
   resource-matched learned control.
4. **Acquisition:** interactions needed to reach `90%` private-query accuracy
   must be at most half the strongest control's count under the same probe
   budget.
5. **Retention:** after learning or changing another rule, every protected rule
   may lose at most one absolute accuracy point outside its paired confidence
   interval; median forgetting must be nonpositive.
6. **Causal state:** constant and independent-environment state must fall to the
   theorem/task ceiling. A within-lifetime shuffle is not accepted.
7. **Horizon/surface:** all above gates hold at `>=2x` training horizon and on
   unseen symbol/serialization permutations.
8. **Every rotation:** no averaging can rescue a failed family.

The decisive run uses at least eight predeclared seeds. Report every seed and a
paired bootstrap `95%` confidence interval. The regret and acquisition gates
pass only if the interval supports the threshold, not merely the point
estimate. Development seeds and decisive seeds are disjoint.

## 7. Resource ledger

Record for every candidate and learned control:

- static parameters and checkpoint bytes;
- persistent state, metadata, copies, and served precision;
- current context and KV bytes;
- forward, write, retrieval, backward, and optimizer FLOPs/work;
- p50/p95 update and prediction latency, throughput, bandwidth, and peak memory;
- environment transitions, probes, generated/scored tokens, simulator calls;
- offline training FLOPs, wall time, hyperparameter trials, and energy where
  available; and
- amortized cost and break-even lifetime count.

The local run reports what can be measured on the local device and marks
production-only operational costs—tenant isolation, encryption, migration,
eviction, corruption recovery, deletion, reset—as unmeasured rather than zero.

## 8. Execution and stop rules

1. Run theorem/channel-integrity unit tests first.
2. Run one smoke seed and verify no family, seed, transcript, or state leak.
3. Freeze code, manifest, thresholds, and analysis hashes.
4. Run the eight decisive seeds locally only.
5. Stop on oracle failure, leakage, control-budget unfairness, or a tie by the
   strongest bounded learned control.
6. Do not tune after decisive results and do not rescue a failed family.

The decisive local budget is capped before execution; it must remain a cheap
screen. No Vast.ai or RunPod instance may be created for T70 at this stage.
Local GPU availability is checked only after this preregistration and integrity
review. If local GPU is busy, use CPU if the cap remains practical or wait.

## 9. Interpretation ladder

- Unit-test pass only: state transport works; no intelligence conclusion.
- All rotated synthetic gates pass: transferable bounded online-learning skill;
  authorize a natural-domain pilot.
- Natural pilot: use noisy/delayed language or tool feedback from a domain held
  entirely out of updater training; test tenant-state lifecycle and compare a
  larger frozen model with equal evidence access.
- Only a replicated natural win with the complete cost ledger can trigger a
  rented or scaled experiment.

No synthetic outcome alone supports “production intelligence.”
