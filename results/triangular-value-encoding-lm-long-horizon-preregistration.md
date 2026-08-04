# TVE 100M-token long-horizon falsification

Two 10M-token runs have shown a nearly identical TVE advantage over the gauge-matched control: 0.269461 percent at exploratory seed 3307 and 0.270290 percent at untouched replication seed 4409. The untouched replication gain decreased from 0.435783 percent at step 61 to 0.270290 percent at step 305. This experiment is frozen specifically to test whether the effect is transient optimization acceleration that disappears when the control catches up.

## Frozen protocol

- Architecture: the same 37,758,336-parameter, hidden-384, 12-layer, 6-query-head, 2-KV-head, head-dimension-64 model.
- Arms: packed raw control, gauge-canonical value control, and triangular value encoding.
- Data: the pinned 100M-token FineWeb-Edu file and the same document-disjoint validation file described by `self-product-ffn-scale-data-manifest.json`.
- Untouched seeds: 5501, 6607, and 7717.
- Per arm and seed: 3,050 steps, microbatch 32, accumulation 2, sequence length 512, 99,942,400 prediction tokens, AdamW, learning rate 3e-4, weight decay 0.1, 50 warmup steps, cosine decay, and gradient clipping at 1.0.
- Evaluations: steps 305, 763, 1,526, and 3,050, corresponding to approximately 10M, 25M, 50M, and 100M prediction tokens; 64 fixed paired validation batches of 32 sequences each.
- The H100 v6 serving gate, architecture constants (`block_size=16`, `tau=0.125`), runtime versions, all executable local dependencies, data ledger, and this preregistration are hash-bound.

## Primary decision

At 100M tokens, compute the candidate-minus-canonical mean loss for each seed, then form a two-sided 95 percent Student-t interval across the three seed means (`t_0.975,df=2 = 4.302652729911275`). Pass requires:

1. all three seed means favor TVE;
2. the seed-clustered upper bound clears an improvement of 0.01 percent of mean canonical loss;
3. the seed-clustered terminal full-minus-disabled upper bound clears 0.005 percent of mean TVE loss.

The trajectory at 10M, 25M, 50M, and 100M is reported without selecting a favorable checkpoint. The 100M endpoint alone is primary.

## Invariants

All arms must have equal parameter counts, parameter tensor counts, buffer counts, serialized state bytes, and optimizer-state bytes. Candidate and canonical initial evaluation must be bit-exact for every seed. Every trained candidate must retain bounded nonzero BF16 coefficients and a bit-exact 12-layer serving bridge. All checkpoint and BF16-export paths, sizes, and SHA-256 hashes must verify. H100 v6 must remain fully passing. Training time and peak memory are reported but are not serving-cost gates.

Passing would rule out simple 10M-token control catch-up at this model scale. It would still require larger-model and second-corpus confirmation before a general breakthrough claim.
