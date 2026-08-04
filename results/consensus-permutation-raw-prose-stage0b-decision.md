# Consensus-permutation raw-prose compiler — corrected Stage 0b decision

Status: all 12 corrected gates passed; admit representation integration  
Date: 2026-07-31

## Valid result

The corrected compiler recovered all 768 direct facts, 384 aliases, and 36
relation paraphrases from 290,304 raw prose strings containing 15.03%
contradictions.  It answered all 4,608 held-out categorical equality queries by
comparing recovered opaque value strings directly.

Unlike the invalidated XOR metric, equality needs no hidden mapping from value
words to integers.  It is invariant under arbitrary renaming of all 16 value
categories.  The compiler API accepts only `Sequence[str]`.

All rejection and accounting gates also passed:

- independent view tables produced zero admissible alignments;
- duplicate complete entity signatures were rejected as non-identifiable;
- the registered whole-corpus majority-error bound was `9.12e-4`;
- the extraction took 12.63 CPU seconds and zero GPU seconds;
- relation alignment tested 72 histogram matches/search candidates.

## Decision

This closes the first missing information link in a controlled regime:
redundant prose can be converted into an exact schema without a teacher or
latent IDs.  It admits the next test only.

The next test must compile the recovered string table into the existing
Transformer, retrieve facts through every alias/paraphrase surface, and execute
equality inside its ordinary attention/SwiGLU graph.  A pass there still does
not prove better training: matched Muon/AdamW controls must then receive both
the same raw corpus and the same compiler-derived labels.
