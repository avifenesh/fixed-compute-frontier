# Reflex-SwiGLU matched language-model screen preregistration

Status: frozen after data preparation and a two-step execution smoke, before the preregistered 640-step screen. The smoke is excluded from quality evidence. Audit-driven amendments added ledger enforcement, stability gates, paired intervals, and eager execution after the compiled baseline failed before its first optimizer step. Data-manifest SHA-256: `955cc3d3be7992840da95881450e4483b2dee4a9567b062b17e1ebe061c817a4`.

## Question

At the same matrix dimensions, checkpoint, token stream, trainable parameter count, and optimizer schedule, does a local self-feedback activation produce lower terminal language-model loss than ordinary SwiGLU and simpler uses of the same added parameter vector?

This is an upcycling screen with no corpus difference between arms. Every arm starts from the exact same released pretrained checkpoint and consumes byte-identical batches in the same order. SmolLM2 was pretrained on FineWeb-Edu, so this validation split is document-disjoint within the continuation run but is not guaranteed held out from the released checkpoint. A pass measures continuation-learning efficiency on a familiar source distribution; it is not yet general capability evidence.

## Frozen model and data

- Model: `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Architecture: 30 Llama layers, `D=576`, `M=1536`, 9 query heads, 3 KV heads.
- Dataset: `HuggingFaceTB/smollm-corpus`, configuration `fineweb-edu-dedup`, revision `3ba9d605774198c5868892d7a8deda78031a781f`.
- Documents are assigned before tokenization by `sha256(id) mod 10`: bucket 0 is validation; buckets 1-9 are training. There is no document overlap.
- Tokenizer is loaded from the frozen model revision. Documents are packed with EOS and no BOS.
- Sequence length 512. Each stored sequence contains 513 tokens for 512 next-token targets.
- Training: 40,960 sequences = 20,971,520 prediction tokens.
- Validation: 4,096 sequences = 2,097,152 prediction tokens.
- The token-file manifest must include SHA-256 hashes.

## Frozen arms

Every Llama MLP receives one trainable `beta` vector of length `M`, including the baseline. Let `a = 2 tanh(beta/2)` and `z0 = SiLU(g) * u`.

1. `baseline`: `z = z0 + 0*a`; beta has an exactly zero gradient and optimizer state but no functional effect.
2. `reflex`: `z = SiLU(g + a clip(z0,-1,1)) * u`.
3. `reflex_detach`: same forward function, but the clipped feedback value is detached in backpropagation.
4. `gate_bias`: `z = SiLU(g + a) * u`.
5. `gate_temperature`: `z = SiLU(g * (1 + 0.5 a)) * u`.

All arms are exactly ordinary SwiGLU at initialization. The last two test whether a gain is explained by two simple per-channel activation parameterizations. They do not isolate `z0` feedback from every possible input-dependent local modulator; a passing Reflex arm must later face `clip(u)`, `clip(SiLU(g))`, and postactivation-feedback controls before a causal self-feedback claim.

## Frozen optimization

- One H100 SXM 80 GB; PyTorch 2.5.1+cu124; Transformers 4.57.6.
- FP32 parameters and AdamW state, BF16 autocast compute, SDPA attention.
- Eager training execution. A pre-run `torch.compile` smoke was rejected because the unchanged baseline produced a non-finite first-step gradient norm; no candidate quality result was observed.
- Microbatch 32, gradient accumulation 2, global batch 32,768 target tokens.
- 640 optimizer steps. Evaluation at steps 0, 40, 160, and 640.
- AdamW betas `(0.9,0.95)`, epsilon `1e-8`.
- Shared weights: peak LR `1e-4`, weight decay `0.1`.
- Beta vectors: peak LR `5e-4`, zero weight decay.
- 32-step linear warmup followed by cosine decay to 10% of peak LR.
- Global gradient clipping at 1.0. No dropout and no gradient checkpointing.
- Seed 101 is reset before every arm. Batches are read sequentially from the frozen token file.

## Required validity checks

1. All five arms have identical total trainable parameter counts.
2. Step-0 validation losses differ from baseline by at most `1e-7`.
3. Every arm completes the exact token budget with finite loss.
4. No arm has a non-finite gradient norm; record maximum pre-clip norm and loss.
5. Record terminal alpha mean/max, fraction `|alpha|>1.9`, clip occupancy, value-activation tails, tokens/s, and peak allocated GPU memory.
6. Parse the data manifest and verify both token-file SHA-256 hashes, byte counts, sequence counts, revisions, sequence length, and document-disjoint split rule before loading the model.
7. Parse the fused H100 result and enforce its validity plus the preregistered `<=1.02x` batch-1/batch-8 serving ratios in the final decision.
8. Assert the exact arms, device, software versions, batch geometry, step/evaluation schedule, learning rates, token budget, eager backend, and saved-checkpoint setting before a full run.
9. Reject an arm if its maximum pre-clip gradient norm exceeds 100, its maximum recorded training loss exceeds 20, or more than 1% of effective alpha values end with `|alpha|>1.9`.
10. Record beta-gradient statistics at step 1 as well as evaluation checkpoints, and record the fraction of optimizer steps whose pre-clip norm exceeds 1.0.
11. Use paired validation-batch differences; the upper endpoint of a normal 95% interval must also favor the selected Reflex arm over baseline and the best simple control.

## Frozen decision rule

Let `R` be the better terminal validation loss of `reflex` and `reflex_detach`; let `C` be the better of `gate_bias` and `gate_temperature`.

The Reflex family advances to a longer three-seed test only if all conditions hold:

1. `R` is at least 0.2% lower than the baseline terminal loss.
2. `R` is at least 0.1% lower than `C`.
3. Its advantage does not exist only at step 40; it is present at both steps 160 and 640.
4. Stability checks pass and the already-measured target-scale (`D=4096,M=14336`) fused serving ratio remains at most 1.02x on batches 1 and 8. This is executor evidence for the algebra, not a latency measurement of the 135M training model (`D=576,M=1536`).

This one-stream screen cannot establish a breakthrough. Failure closes this exact retrofit protocol, not the mathematical function class; because endpoint gradient scales differ between controls, failure first triggers an LR/conditioning diagnosis. Revival would require a materially different, preregistered training-from-scratch or parameterization hypothesis. Passing authorizes a longer confirmation using three independently hashed document subsets/orders—not merely RNG seeds—plus a post-checkpoint external validation corpus and token-matched and wall-clock-matched reporting. The better Reflex variant is locked before confirmation to avoid selecting it again on confirmation data.
