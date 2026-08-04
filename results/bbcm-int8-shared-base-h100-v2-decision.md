# BBCM INT8 shared-base H100 v2 decision

Decision: pass as an exact-stored-scale checkpoint endpoint.

- BF16 baseline NLL: `2.8295718897134066`.
- INT8 decoded NLL: `2.833453081548214`.
- Relative NLL degradation: `0.1371653376%`.
- Paired absolute-NLL 95% interval: `[0.0037306242, 0.0040317595]`.
- Every scale is exactly BF16-representable; all 90 FFN matrices were quantized.
- Result SHA-256: `681ac7d8ca088bed135f33752758ef985e8d71ec54f0ec9ce4ca548b7e3dec72`.

This authorizes only the exact ternary-residual endpoint test. It does not establish equal width, routing, packing, or runtime.
