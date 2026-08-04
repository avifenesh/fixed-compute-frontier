# Raw-prose equality plane T11 — matched-training pilot protocol

Status: frozen before pilot execution  
Date: 2026-07-31

T11 reuses the complete T10 matched-training protocol without changing its
data split, controls, optimizer, schedule, checkpoints, or gates.  The only
candidate-side change is the preregistered minimal triangular writer in
`raw_prose_equality_plane_t11.py`.

## Arms

- `compiler_muon_1x`: T11 direct writer, no gradient prefix, 1,000 mixed steps;
- `muon_labels_1x`: one raw/direct/equality prefix pass, 1,000 mixed steps;
- `muon_labels_2x`: two prefix passes, 2,000 mixed steps;
- `adamw_labels_2x`: two prefix passes, 2,000 mixed steps.

Muon uses learning rate 0.005 and AdamW uses 0.0003.  Every twentieth mixed
step is a knowledge step, alternating direct and equality batches; all other
steps draw from the protected document-disjoint natural stream.

## Frozen gates

1. The T11 exhaustive BF16 representation artifact passes first.
2. Candidate held-out direct and equality accuracy is exactly 100%, with a
   positive minimum margin, at steps 250, 500, and 1,000.
3. Candidate natural NLL is at most 1.005 times `muon_labels_1x` NLL at every
   matched checkpoint.
4. The best twice-trained control remains below 90% on either held-out direct
   or held-out equality.
5. Initial model hashes are identical, all values are finite, and every data,
   update, time, memory, and integrity ledger is emitted.

Passing admits three sealed seeds.  It is not the project endpoint and cannot
be described as a smarter production LLM.
