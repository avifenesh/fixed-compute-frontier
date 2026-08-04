# TVE H100 transitive-dependency replication v6

V6 repeats the successful v5 serving gate at a fresh seed and closes its remaining integrity gap by binding the transitive value-flow source that defines the canonical RQ transform. The executor helper, attention implementation, three focused tests, and this preregistration remain bound as well.

The executor, arithmetic, shapes, rows, 30 warmups, 200 randomized paired trials, arithmetic 128 MiB L2 touch, and 5,000 bootstrap replicates are unchanged from v5.

Every cell must retain: bit-exact control and serving-aligned candidate outputs; a nonzero candidate path; PyTorch-algebra drift below 0.015625 max, 2e-8 mean, and 1e-5 fraction; propagated O drift below 0.015625 max and 1e-6 mean; proxy-NLL drift below 1e-5; latency upper 95 percent ratio at most 1.025; and zero added learned state, KV bytes, or runtime metadata.

`block_size=16` and `tau=0.125` are fixed architecture/compile constants, like head dimension, rather than per-layer or per-token runtime metadata.
