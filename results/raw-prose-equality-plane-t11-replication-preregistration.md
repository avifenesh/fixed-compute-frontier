# Raw-prose equality plane T11 — unseen-seed replication preregistration

Status: frozen before replication execution  
Date: 2026-07-31

## Purpose

Test whether the positive T11 pilot survives model initialization and batch
sampling changes.  The pilot seed 6,401 is excluded.  The sealed unseen seeds
are 6,703, 6,997, and 7,307.

## Frozen protocol

Each seed independently reruns all four T11 pilot arms:

- `compiler_muon_1x`;
- `muon_labels_1x`;
- `muon_labels_2x`;
- `adamw_labels_2x`.

The writer, output scale, raw corpus, query split, natural train/validation
files, arm order, optimizer settings, schedules, token presentations,
checkpoint evaluations, thresholds, and resource ledgers are byte-identical
to the passed T11 pilot harness.  No seed may reuse a trained checkpoint.

## Frozen gates

The replication passes only if all three unseen seeds independently satisfy
every T11 pilot gate:

1. exact held-out direct and equality accuracy with positive BF16 margins at
   steps 250, 500, and 1,000;
2. candidate natural NLL no more than 0.5% above the matched `muon_labels_1x`
   control at all three checkpoints;
3. the best twice-trained control below 90% on either held-out capability;
4. identical within-seed initial hashes, finite training, no failed or retried
   updates, and complete resource/integrity ledgers.

The aggregate artifact must report every per-seed result, the worst natural
NLL delta, and the lowest terminal direct/equality margin.  There is no
averaging escape: one failed seed fails replication.

## Decision rule

A pass establishes reproducibility only within this controlled scale.  It
admits a separately preregistered real-prose/unrestricted-QA scale test.  It
does not establish a smarter production LLM.

A failure closes T11 as a robust method at this scale.  No seed deletion,
margin-floor relaxation, or post-result mask/output-scale change belongs to
this replication.
