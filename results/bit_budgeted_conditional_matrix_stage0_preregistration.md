# Bit-Budgeted Conditional Matrix (BBCM) stage-0 preregistration

Status: frozen before running the arithmetic witness.

## Candidate algebra

Replace each dense BF16 weight with a context-indexed decoded weight:

```
W_e = dequant8(Q0) + dequant2(Delta[e])
e   = top1(router(x))
y   = W_e x
```

`Q0` is an 8-bit shared base and each of four routes has a 2-bit delta. Per original matrix coordinate, `8 + 4*2 = 16` stored payload bits: exactly one BF16 word. Only the selected decoded matrix participates in the GEMM. This is a weight algebra, not an additive shared-expert path.

For a target SwiGLU FFN with `D=4096, M=14336`, the exact-storage implementation may reduce `M` to an aligned value to fund BF16 per-output scales and a BF16 `D x 4` router.

## Frozen stage-0 gates

1. Total candidate resident bytes, including scales and router, do not exceed the BF16 FFN matrix bytes.
2. Candidate FFN matrix MACs plus router MACs per token do not exceed baseline FFN matrix MACs.
3. The bit allocation yields the natural cap `K=floor((16-8)/2)=4` experts.
4. A frozen scalar piecewise-slope witness is represented exactly by four routed decoded weights but not by the best single affine map.
5. The candidate and one BF16 word have the same raw code cardinality, `2^16`; the gain is conditional factorization, not extra information bits.

Passing proves only an exact resource/function-class ledger. It does not prove useful language specialization, quantization tolerance, routability, or GPU efficiency.

## Prior-art boundary

- MoTE trains a full-precision shared FFN plus a separately evaluated ternary routed FFN. BBCM instead decodes one selected composite matrix and performs one FFN.
- MoQE quantizes already-expanded experts; it does not require an exact dense-model bit budget.
- Delta-compression methods share a base and compress expert differences, but generally start from an existing MoE and may add decompression or low-rank execution.

The potentially distinct hypothesis is therefore narrow: **train a context-indexed base-plus-bitplane matrix under an exact dense BF16 resident-bit and active-MAC ledger, and decode it inside one routed GEMM.** No novelty claim is made until a broader collision audit.

