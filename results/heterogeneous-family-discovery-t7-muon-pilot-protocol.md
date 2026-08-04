# T7 Muon calibration pilot protocol

Frozen before measurement: 2026-07-31

This is optimizer calibration, not evidence for the heterogeneous-compilation
claim.  It uses one development initialization and natural FineWeb-Edu batches
only.  It may select the control recipe but cannot be included among the three
sealed model seeds or the untouched capability evaluations.

## Fixed comparison

- Model: the 110,776,960-parameter T7 causal LM (width 640, 16 blocks, ten
  64-wide heads, SwiGLU width 1,728, tied vocabulary 49,152).
- Data: identical training batches, initialization, validation batches, BF16
  forward arithmetic, FP32 loss, clip norm 1, and warmup/cosine schedule.
- Reference: AdamW at `3e-4` on all parameters.
- Muon arms: matrix peak learning rates `0.005`, `0.01`, `0.02`, and `0.04`.
  Hidden-layer matrices use five-step Newton--Schulz Muon with
  `match_rms_adamw`; the embedding and RMSNorm vectors use AdamW at `3e-4`.
- Every arm trains for 250 steps.  Validation is recorded at steps 50, 100,
  and 250 on the same 16 batches.

The selected Muon rate is the finite Muon arm with the lowest step-250 held-out
NLL.  Ties go to the lower learning rate.  Maximum loss and pre-clip gradient
norm, wall/GPU time, sampled power, HBM, optimizer state, mathematical training
FLOPs, and Muon optimizer FLOPs are recorded.  No capability examples or T7
model seeds are exposed during calibration.

