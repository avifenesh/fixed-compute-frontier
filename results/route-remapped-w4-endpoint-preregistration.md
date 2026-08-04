# Route-remapped W4 fatal endpoint

## Question

Can a single shared W4 symbol matrix with BF16 group-128 scales preserve the
released SmolLM2-135M FFN endpoint closely enough to serve as the substrate for
route-specific scale/LUT decoders?

This test contains no routing and makes no runtime or novelty claim.  If the
shared codes already destroy the endpoint, conditional remapping cannot be a
fixed-cost improvement without first paying to recover the base model.

## Frozen protocol

- Model: `HuggingFaceTB/SmolLM2-135M` revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Device/runtime: the retained H100, PyTorch `2.5.1+cu124`, CUDA `12.4`,
  Transformers `4.57.6`.
- Data: the hash-bound 4,096-sequence validation set from
  `results/reflex-swiglu-lm-data-manifest.json`.
- Evaluation: 128 batches of 32 sequences, length 512.
- Seed: 131.  Input, manifest, and output paths are frozen to the repository
  defaults; command-line path substitution is protocol drift.
- Quantize all 90 FFN gate/up/down matrices.  Along each output row, split the
  input axis into consecutive groups of at most 128 values.  Store one signed
  4-bit code in `[-7,7]` per weight and one BF16 symmetric scale per group.
  The shorter final group is charged identically and no padding code is stored.
- Decode to FP32 only for this quality endpoint.  Physical packing and kernel
  speed are outside this gate.

## Frozen gate

The branch advances only if:

1. the BF16 baseline matches `2.8295718897134066` within `1e-7`;
2. relative decoded NLL degradation is at most `0.2%`;
3. the paired candidate-minus-baseline 95% interval upper bound is at most
   `0.002 * baseline NLL`;
4. all 90 matrices, counts, scales, codes, losses, and errors are finite; every
   code is an integer in `[-7,7]`, group counts are exact, and all stored scales
   round-trip exactly through BF16;
5. the exact resident ledger is reported separately from active arithmetic.

A pass authorizes only a held-out teacher-output reconstruction test of
scale-only and global-LUT route decoders against static equal-bit and K3
independent-W4 controls.  A failure closes this codebook substrate before any
routing, learning, or custom-kernel work.
