# TVE LM confirmatory replication

This is the untouched-seed replication. It explicitly follows an exploratory one-step smoke and a favorable seed-3307 10M-token discovery. No claim rests on the discovery alone.

The architecture, three arms, data, 12-layer hidden-384 model, 9,994,240 tokens per arm, optimizer, evaluation batches, and decision thresholds are inherited unchanged from the discovery: candidate/control exact initialization; canonical/raw initial drift within 0.02 percent; candidate paired upper bounds improve over raw and canonical by at least 0.025 percent; encoding ablation clears 0.01 percent; equal budgets; bounded BF16 coefficients; and bit-exact trained serving export.

The only statistical change is the untouched initialization/training seed 4409. The same data is intentionally retained to isolate initialization variance.

This replication additionally requires:

- exact train, validation, data-manifest, output, and checkpoint paths;
- Triton 3.1.0 and every direct executable dependency hash;
- the fully transitive-dependency-bound H100 v6 gate;
- all three checkpoint hashes and the candidate BF16 attention-export hash and bytes;
- fixed architecture constants block size 16 and tau 0.125, with zero per-layer/per-token runtime metadata.

Passing establishes a replicated small-scale result, not production-scale universality. The next gate would be a larger model/token scale and a second corpus.
