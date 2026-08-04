# BBCM ternary-residual endpoint H100 v1 decision

Decision: invalid as an exact-format gate.

The measured endpoint remains a useful FP32-scale diagnostic, but both base and delta scales were retained in FP32 while the storage ledger funds BF16. It cannot authorize escalation. V2 uses BF16-stored scales throughout quantization and decoding.
