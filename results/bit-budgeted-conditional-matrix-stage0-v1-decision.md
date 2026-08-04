# BBCM stage-0 v1 decision

Decision: **invalid; withdrawn before use as an architecture gate.**

The storage and MAC arithmetic was correct, but the artificial function witness hard-coded four scalar intervals while the counted router was bias-free linear. Along `x=t*v`, a bias-free top-1 linear router can select at most two experts. The witness also reused its target route function for prediction instead of instantiating router parameters and base/delta codes.

The v1 JSON remains as an audit artifact. Stage 0 must be rerun with a counted affine router and explicit code decoding.

