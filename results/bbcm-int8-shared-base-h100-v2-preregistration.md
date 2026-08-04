# BBCM INT8 shared-base H100 quality gate v2

Status: frozen after invalidating v1's scale format, before v2 execution.

## Repair

V1 computed and retained per-output scales in FP32 even though the exact storage ledger funds BF16 scales. V2 rounds every scale to BF16 before assigning INT8 codes and before decoding. Zero rows store the exactly representable BF16 value `1` and decode exactly to zero.

Everything else is frozen to v1: SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, manifest SHA-256 `955cc3d3be7992840da95881450e4483b2dee4a9567b062b17e1ebe061c817a4`, 4,096 validation sequences of length 512, batch 32, all 90 FFN matrices, H100, PyTorch 2.5.1+cu124, CUDA 12.4, Transformers 4.57.6, and frozen baseline NLL `2.8295718903541565`.

## Frozen gates

1. Exact environment, data, model, and baseline anchor.
2. INT8 decoded FFN relative NLL degradation is at most 0.2%.
3. All 90 matrices and statistics are finite.
4. Every stored scale is exactly BF16-representable.

Passing establishes the exact stored-scale checkpoint endpoint only. It does not establish routed specialization, bit packing, or equal latency.
