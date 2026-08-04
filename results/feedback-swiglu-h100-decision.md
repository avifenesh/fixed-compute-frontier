# Feedback-SwiGLU stock H100 executor decision

Status: **stock `torch.compile` executor rejected; fused executor remains authorized**.

The frozen H100 protocol was valid, source and preregistration hashes matched, and zero-feedback inclusion was exact. All latency gates failed:

| Tokens | Same width / baseline | Equal parameter / baseline |
|---:|---:|---:|
| 1 | 1.0430x | 1.0481x |
| 8 | 1.0422x | 1.0449x |
| 32 | 1.0434x | 1.0450x |
| 128 | 1.0561x | 1.0587x |

The 4.2% to 5.9% cost is much larger than the 0.13% matrix-MAC ledger predicts. The likely causes are two extra small matrix launches, intermediate activation traffic, and a second activation pass. Per the preregistration, the next valid test must fuse squeeze, group-local rank-8 expansion, and regating into one kernel. More samples or threshold changes are not justified.

The single-GPU result does not test the tensor-parallel claim. No-new-collective deployment requires whole-group placement; with 8 equal groups this cleanly supports balanced TP degrees 1, 2, 4, and 8.

