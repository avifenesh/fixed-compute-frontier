# T84 effective-v4 preregistration acceptance

Date: 2026-08-02  
Decision: **ACCEPTED FOR SOURCE IMPLEMENTATION ONLY**  
Execution authorization: **NONE**

The accepted preregistration is the immutable ordered triple:

1. `shared-score-heterogeneous-attention-t84-cpu-preregistration-v2.md`  
   SHA-256 `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`
2. `shared-score-heterogeneous-attention-t84-cpu-preregistration-v3.md`  
   SHA-256 `b10c376e386c916bfb25bc11a0af2ef3f2433439b99d98c5e5515af2047b6bbf`
3. `shared-score-heterogeneous-attention-t84-cpu-preregistration-v4.md`  
   SHA-256 `085abc5e703ee4a064a837f66cf3ffb82e282c1fe78fec12cc53870a0db774b8`

Both independent auditors accepted that exact ordered triple after separately
rejecting v1, v2, and one side of v3:

- audit A SHA-256
  `ab87d477e1300986591af0a30fd3024fd716cc4464226629150186827d0356ce`;
  final verdict `ACCEPT EFFECTIVE V4`;
- audit B SHA-256
  `2ae7b68e21972596b3cb0297a2fc6a9ca2e38e5283f19343f74cd6b45df7915c`;
  final verdict `ACCEPT EFFECTIVE V4`.

Bound inputs are:

- mixer text SHA-256
  `f84aa219820c84314da2eede2e0e8b45bc83c881a1b01d2b1d6973d1cd8537c8`;
- canonical mixer binary SHA-256
  `2c1ac147b8f896e889695097fa732e2811ec8aeabe45fddc211482f08a45c973`;
- provenance-wrapped Tropical Attention snapshot SHA-256
  `af11263442321708748c06cb41c1493bf8da9f15e1e6099aaa293008ed03ef51`;
- stripped upstream file SHA-256
  `5d06665382adc632d63028f7eab3b1c62e84ca183c6d0fb594813da8a19a7068`;
- upstream commit
  `e3c12f3e7c401245b9b5577d7181922b0150efc6`.

Acceptance permits writing source, tests, decoder, and static manifests without
executing them. Fixture generation and timing require a later independent
source audit returning `FIXTURES AND TIMING`. Stage 0 requires the subsequent
artifact audit. No CPU experiment, GPU work, or rental is authorized here.
