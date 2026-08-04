# BBCM INT8 shared-base H100 v1 decision

Decision: invalid as an exact-format gate.

The measured NLL result remains a useful FP32-scale diagnostic, but v1 retained scales in FP32 while the storage ledger funds BF16. It cannot authorize escalation. V2 rounds stored scales to BF16 before code assignment and decoding.
