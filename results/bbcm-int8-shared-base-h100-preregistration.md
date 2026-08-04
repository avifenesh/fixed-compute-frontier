# BBCM INT8 shared-base H100 quality gate

Status: frozen before execution.

## Question

Can the released SmolLM2-135M FFN matrices be replaced by per-output-channel symmetric INT8 decoded weights without consuming more than 0.2% relative validation NLL? This tests whether an 8-bit shared base can preserve the pretrained endpoint well enough to fund four 2-bit routed deltas later.

It does not instantiate routing, deltas, bit packing, or a fused kernel.

## Frozen protocol

- Exact model and revision: `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Exact document-disjoint validation file from manifest SHA-256 `955cc3d3be7992840da95881450e4483b2dee4a9567b062b17e1ebe061c817a4`.
- Validation: 4,096 sequences, length 512, 2,097,152 next-token targets; batch 32.
- H100; PyTorch 2.5.1+cu124; CUDA 12.4; Transformers 4.57.6; FP32 resident weights and BF16 autocast evaluation.
- Quantize all gate, up, and down projection weights in all 30 FFNs. Attention, embeddings, norms, and LM head remain unchanged.
- For each output row: `s=max(abs(w))/127`, `q=clip(round(w/s),-127,127)`, `w_hat=s*q`.
- Evaluate baseline first, quantize in place, then evaluate the INT8-decoded base on identical batches.

## Frozen gates

1. Exact environment, data hashes/counts, model architecture, and source/preregistration hashes.
2. Independently measured baseline NLL matches the prior frozen screen endpoint-at-step-0 NLL `2.8295718903541565` within `1e-7`.
3. INT8 decoded FFN relative NLL degradation is at most 0.2%.
4. All losses and quantization statistics are finite; exactly 90 FFN matrices are quantized.

Passing authorizes a routed-delta learning design. It is not evidence that routed deltas improve quality or that decoded weights execute cheaply.

