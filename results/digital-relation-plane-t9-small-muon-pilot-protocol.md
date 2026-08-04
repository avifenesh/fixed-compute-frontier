# T9 36.6M Muon calibration protocol

Frozen before measurement: 2026-07-31

This natural-text-only development pilot calibrates the second-scale optimizer;
it is not capability evidence.  One model seed (5,003), identical initialization
and batches, 250 steps, and held-out NLL at 50, 100, and 250 are used.

AdamW at 3e-4 is the reference.  Hybrid Muon/AdamW matrix learning rates are
0.0025, 0.005, 0.01, and 0.02, with `match_rms_adamw`, five Newton–Schulz
steps, and AdamW 3e-4 for embeddings/norms.  The finite Muon arm with the lowest
step-250 held-out NLL is frozen for all T9 arms.  No T9 model seed or relation
table is exposed.

