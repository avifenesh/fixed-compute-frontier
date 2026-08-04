# TVE direct-native confirmation

## Purpose and disclosed evidence

The frozen three-seed 100M-token long-horizon experiment was a formal NO-GO
because one gauge-canonical control failed its noninferiority bound against the
packed raw transformer. TVE nevertheless beat both controls in all three seeds.
This new experiment does not alter or pool that result. It uses five untouched
seeds and makes the complete TVE model versus the packed raw transformer the
primary endpoint.

## Frozen protocol

- Architecture: 37,758,336 parameters, hidden 384, 12 layers, six query heads,
  two KV heads, head dimension 64, intermediate size 1,024.
- Arms: `packed_raw_control` and `triangular_value_encoding` only.
- Seeds: 8821, 9901, 11003, 12109, and 13217.
- Data: the pinned document-disjoint FineWeb-Edu train and validation files in
  `self-product-ffn-scale-data-manifest.json`.
- Per arm and seed: 3,050 steps, microbatch 32, accumulation 2, sequence length
  512, 99,942,400 prediction tokens, AdamW, learning rate 3e-4, weight decay
  0.1, 50 warmup steps, cosine decay, and gradient clipping at 1.0.
- Evaluations: steps 305, 763, 1,526, and 3,050 over the same 64 fixed paired
  validation batches of 32 sequences.
- TVE constants remain block size 16 and tau 0.125. No new parameters, buffers,
  cache values, or runtime metadata are allowed.
- The prior long-horizon result and decision, H100 v6 serving result, executable
  dependencies, data ledger, source, and this preregistration are hash-bound.

## Primary decision

At 100M tokens, compute the candidate-minus-raw paired mean loss for each of the
five new seeds. Form a two-sided 95 percent Student-t interval over the five seed
means using `t_0.975,df=4 = 2.7764451051977987`.

Pass requires all of the following without pooling prior seeds:

1. all five seed means favor TVE;
2. the clustered upper bound is at most minus 0.01 percent of mean raw loss;
3. mean relative NLL improvement is at least 0.05 percent;
4. the clustered terminal full-minus-disabled upper bound is at most minus
   0.005 percent of mean candidate loss.

The full 10M, 25M, 50M, and 100M trajectories are reported. Only the 100M
endpoint is primary.

## Invariants

- Both arms have identical parameter counts, parameter tensor counts, buffer
  counts, serialized state bytes, optimizer-state bytes, and zero metadata.
- Candidate and raw initial paired NLL intervals must fit within plus or minus
  0.02 percent of raw initial NLL for every seed.
- Candidate coefficients must be finite, nonzero after BF16 export, bounded by
  0.5 in absolute value, export bit-exactly, and retain an all-layer bit-exact
  serving bridge.
- All checkpoint and candidate BF16-export paths, sizes, and SHA-256 hashes must
  verify.
- H100 v6 and all bound integrity checks must remain valid.

Passing would confirm a persistent complete-model gain over the native raw
transformer at one model scale and corpus. It would not establish generality;
the next required gate remains an exact larger model on a second corpus plus a
full-attention native serving-cost comparison.
