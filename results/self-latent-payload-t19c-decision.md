# Self-latent payload T19c — decision

Status: **FAIL; close mean-pooled self-latent payload**  
Date: 2026-07-31

## Decision

Do not implement the alternating in-place writer/reader from this payload.
Mean-pooled final states from the sealed raw-trained model are not a useful
typed document representation for held-out comparison reasoning.  Do not
rescue this result with another layer, pooling rule, projection, context,
classifier, or additional training.

## Result

The fixed reader fit all 146 training labels but scored 52/104 = **50.0%** on
development.  It failed all three capability gates:

- below the required 75%;
- 10.5769 points below T12's semantic probe, rather than ten points above;
- 10.5769 points below T19b's lexical payload, rather than ten points above.

The failure is not numerical or structural:

- the writer checkpoint hash and raw-only metadata matched exactly;
- all 2,405 document and 250 question vectors were finite/nonzero;
- minimum BF16 payload cosine was 0.9999965;
- every question matched exactly two titles without support annotations;
- the exact 743,670-write ordinary-Transformer layout passed again;
- writer, candidate, and control retained the identical model class, state
  schema, and 36,577,152 parameters.

## Interpretation

The same-model writer removes the objection that an external compiler is doing
the language understanding.  Its failure establishes a different boundary:
next-token-trained hidden states do not expose a task-stable typed schema under
a fixed generic pooling map.  Their information may exist in a distributed,
position-dependent form, but copying one pooled vector per document does not
make that information composable.

Together, T19a–T19c now separate the boundaries:

1. **Address:** almost solved, but gradient training also solves it; only a
   small warm-start gain.
2. **Physical payload:** 2,405 records of width 220 fit exactly under the
   demonstrated standard-Transformer write budget.
3. **Lexical payload:** numerically faithful but only 60.5769% useful.
4. **Generic self-latent payload:** numerically faithful but chance-level
   useful.

The remaining missing object is identifiable typed relational structure.  A
successor must explain where that structure comes from without importing a
stronger pretrained semantic model.  Merely changing the vector extraction is
not admissible.

## Evidence

- Result SHA-256:
  `e6776f3f408a9477066c5ddf6be7d75492e2f8938225c64c6959db3d0f961c67`.
- Source SHA-256:
  `6238304d7d4f87dee193b506aecab73c524c5d0d3cd1423a9d04e741b0d4ca21`.
- Preregistration SHA-256:
  `8d65aeb0427460afc29d9b471e16c041d0aaa5c39ecd20bd91e8b2a6e4389e7f`.
