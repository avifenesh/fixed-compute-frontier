# Peak-lifted attention Stage-0 run note

The first remote invocation on 2026-07-27 used the reviewed source and
preregistration hashes, but copied both files into the remote workspace root.
The program expected the preregistration under `results/` and terminated with
`FileNotFoundError` before serializing or exposing a formal result. No code,
threshold, seed, or preregistration changed. The formal run is the next
invocation with the same bytes in their expected directory layout.
