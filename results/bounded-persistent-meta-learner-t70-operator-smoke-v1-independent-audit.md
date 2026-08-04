# Independent audit — T70 operator smoke v1

Date: 2026-08-02  
Audited artifacts:

- `experiments/bounded_persistent_meta_learner_t70_smoke.py`
- `results/bounded-persistent-meta-learner-t70-operator-smoke-v1.json`
- `results/bounded-persistent-meta-learner-t70-operator-smoke-v1-decision.md`

No audited artifact was modified and no experiment was rerun for this audit.

## Verdict

**RETAIN THE FAIL DECISION; CLOSE CROSS-BOUNDARY BPTT NECESSITY ON THIS LOOKUP WITNESS; NO FURTHER LOCAL TEST BEFORE THE FULL ROTATED MANIFEST.**

The causal state channel is valid, the boundary detach does remove the cross-boundary autograd path, and the result arithmetic is correct. The decision is also right that 92.81% detached accuracy falsifies the preregistered claim that query loss must train the observation-time write through the boundary for this lookup task. It does not close lifetime training, learned compression, or T70 across families.

Two wording corrections are required. The “independent shuffle” is a random within-batch permutation that allows fixed points, and the detached observation encoder is not wholly frozen because it shares transformer parameters with the trained query path. The random-code explanation is plausible and sufficient as a postmortem, but the current metrics do not uniquely prove it.

## Code semantics

### Causal write and hard boundary — correct

`forward_segment` orders tokens as two state reads, key, value/missing-value, then two write queries (`smoke.py:77-104`). The upper-triangular mask prevents earlier positions from seeing the appended write slots, while each suffix write can see the event. Only `next_state` crosses from the observation segment to the query segment (`smoke.py:147-153`); no key, value, hidden transcript, or KV cache is passed. The query logit is read at the missing-value position, which can attend the carried state and current key but not the query segment's future write slots. This is the intended causal operator smoke.

### Detach — correct direct-credit control, with shared-weight qualification

`written_state.detach()` (`smoke.py:149-152`) preserves the exact forward state and cuts every gradient path from query loss through that tensor into the earlier observation computation. The detached model is initialized identically and sees the same training batches as the candidate because each `train_one` resets the same seed (`smoke.py:142-147, 197-215`). This is a clean test of **direct cross-boundary gradient credit**.

It is not a globally frozen-writer test. The observation and query segments use the same transformer, key embeddings, and some token/type parameters. Query-time gradients update those shared parameters, changing subsequent observation-time codes even though no gradient traverses a particular earlier write. Conversely, the observed-label embeddings, write queries, initial state, observation-phase embedding, and state-normalization write path receive no useful detached observation loss and remain random/untrained or only indirectly affected. The decision should replace “the untrained ... shared Transformer” with: **fixed random label-bearing features pass through an encoder whose shared reader/transformer weights continue to train, yielding decodable finite lookup codes without direct writer credit.**

This qualification does not rescue the failed necessity claim: the declared detach condition achieved 92.81% without the prohibited gradient path.

### Shuffle — not strictly independent

`torch.randperm(batch_size)` (`smoke.py:182-183`) can leave samples in place. With batch size 64, the expected fixed-point fraction is `1/64`; if fixed points are factual and all other states are independent chance, expected accuracy is

\[
\frac1{64}\cdot1+\frac{63}{64}\cdot\frac14=0.26171875,
\]

almost exactly the reported 0.2625. Nonfixed source examples are iid relative to the target, but the intervention as a whole is not independent. Rename the metric `within_batch_permutation_accuracy`, or use a derangement/nonzero cyclic roll in the future frozen manifest. This implementation defect does not undermine the state-channel conclusion; it explains the small elevation above chance.

## Metric and result audit

The JSON contains one seed and `10*64=640` evaluation examples. The reported proportions reconstruct exactly:

- candidate: `640/640 = 1.000000`;
- detached: `594/640 = 0.928125`;
- reset: `167/640 = 0.2609375`;
- permuted state: `168/640 = 0.262500`.

Nominal class chance is `1/4=0.25`; the permutation intervention has the higher fixed-point expectation derived above. The frozen pass rule required candidate accuracy at least 0.90 and all three controls at most `0.25+0.08=0.33` (`smoke.py:255-261`). Detached accuracy violates its ceiling by 59.81 points, so `status: fail` is correct. The decision's “below 33%” should read “at most 33%,” an immaterial boundary wording error here. Reset and permuted-state scores are consistent with chance/fixed-point leakage; formal confidence intervals were not preregistered and are unnecessary for the gross detached failure.

The evaluation correctly reuses the same generated batches across candidate and detached models by reseeding each evaluation (`smoke.py:216-233`). It reports candidate reset/permutation ablations only. Therefore the claim that the **detached** model decodes the carried random code is strongly implied by its factual accuracy and the source path, but not directly isolated. Detached reset/permutation metrics or a separately frozen random encoder would establish the mechanism more cleanly. They are postmortem diagnostics, not prerequisites for accepting the failure.

For future frozen artifacts, include model dimensions/state slots, thread count, evaluation seed, exact success counts, framework version, and source revision in the JSON; those are currently split between hard-coded source and prose.

## Random-code interpretation

The interpretation is correct as an existence explanation, with narrower language:

1. actual value embeddings are random and label-specific;
2. a two-slot, width-16 real state has ample continuous capacity for the finite key/value set;
3. the observation forward pass transports those features into state even without cross-boundary gradient; and
4. the trained query-side decoder can learn the finite state/key-to-value map.

A generic random real-valued map is often injective on a finite set under nondegeneracy, but this is not guaranteed by “continuous randomization,” does not imply robustness at served precision/noise, and says nothing about systematic transfer. The source also does not prove that this is the unique mechanism because the shared transformer changes during detached training. The decision already correctly labels the statement as existence rather than finite-sample/robustness theory; change “stable random codes” to “decodable random-feature codes under the shared trained operator.”

## Decision and next-test admission

The writer-credit claim should close **on lookup**. One successful no-direct-credit construction is enough to refute necessity on this witness, and the 59.81-point threshold violation is not a marginal seed effect. More lookup seeds cannot turn this task into a test of learned abstraction; changing the threshold after observation would be invalid.

Do not insert another local lookup/compression variant before the full rotated manifest. A detached-state shuffle or frozen-random-encoder check could refine the postmortem, but it cannot advance the substantial-intelligence hypothesis and is not needed for the closure. The next admissible experiment is the already-required frozen cross-family/rotated screen, with:

- a strict derangement or independent-environment state intervention;
- full-context oracle competence before memory scoring;
- finite precision/noise and a binding state budget;
- held-out rules/compositions where memorizing a finite random code cannot solve the task;
- detached-forward-state, reset, bounded text/retrieval, and same-resource recurrent controls; and
- a fully frozen manifest, seeds, thresholds, run cap, and resource ledger.

Until that manifest exists, the decision's “no decisive local run and no rental” is correct.
