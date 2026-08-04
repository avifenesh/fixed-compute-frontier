# Triangular value encoding LM discovery preregistration

This freezes the first capability test after the serving-aligned H100 v4 gate passed. It is a one-seed discovery experiment; even a pass requires independent replication.

## Fixed arms

All arms use the same 12-layer, hidden-384, 6-query-head, 2-KV-head, head-dimension-64 Llama-family model and the same packed-QKV, FP32-one-store RoPE execution.

1. `packed_raw_control`: original V/O coordinates; no value encoding.
2. `canonical_value_control`: deterministic sign-canonical RQ of each V/O GQA gauge; no value encoding.
3. `triangular_value_encoding`: identical canonical initialization plus block-16 nonlinear value encoding.

The candidate forward is the exact Triton serving forward. The algebraically equivalent PyTorch expression supplies gradients through a zero-forward straight-through bridge. All arms must have exactly the same learned scalars, parameter tensors, buffers, serialized state, optimizer state, KV dimensions, and metadata.

## Frozen training and evaluation

- H100 80GB, Python 3.11.10, NumPy 2.1.2, Torch 2.5.1+cu124, CUDA 12.4, Transformers 4.57.6.
- Fixed block-algebra scratch train and validation uint16 files and manifest.
- Sequence length 512; microbatch 32; accumulation 2; 305 optimizer steps.
- 9,994,240 prediction tokens per arm.
- AdamW, learning rate 3e-4, betas 0.9 and 0.95, weight decay 0.1, clip 1.0.
- Fifty warmup steps and cosine schedule inherited from the established scratch protocol.
- Seed 3307 for every arm.
- Evaluate at steps 0, 61, and 305 on 64 fixed batches of 32 sequences.
- Save and hash every terminal checkpoint and a BF16 candidate attention export.

## Frozen pass gates

1. Exact parameter count is 37,758,336 for every arm; all parameter/state/optimizer/buffer counts match and metadata is zero.
2. Candidate and canonical control initial validation loss and every per-batch loss are bit exact.
3. Canonical/raw initial mean and paired interval are within 0.02 percent. Canonical terminal is noninferior to raw within 0.05 percent.
4. Candidate terminal paired-loss upper bounds beat both raw and canonical control by at least 0.025 percent of the corresponding terminal loss.
5. Disabling only value encoding in the trained candidate worsens mean loss by at least 0.01 percent, and the paired upper bound clears the same 0.01 percent threshold.
6. All metrics are finite and all value nonfinite fractions are zero.
7. Some coefficient slots survive BF16 export, maximum absolute coefficient is at most 0.5, and coefficient export is bit exact.
8. Every trained candidate layer matches the BF16 cache-serving executor bit for bit.
9. H100 v4 admission, protocol, source hashes, and data ledger remain valid; all formal artifacts are saved and hashed.

The minimum quality effect is more than 100 times larger than the maximum admitted proxy-NLL difference between legal PyTorch and serving reduction orderings.
