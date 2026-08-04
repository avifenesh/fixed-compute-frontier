# Self-latent payload T19c — sufficiency preregistration

Status: **frozen before result**  
Date: 2026-07-31

## Hypothesis

T19b proved that a 220-dimensional per-document payload fits exactly inside
the demonstrated write budget, but a fixed lexical CountSketch reached only
60.5769% comparison accuracy.  T19c asks whether the same from-zero model that
read the raw documents can serve as its own training-only payload compiler.

This avoids the stronger-compiler objection: writer and deployed reader are
the identical 36,577,152-parameter Transformer class.  The writer sees no
question, answer, support annotation, external embedding, pretrained teacher,
or ontology.  It is discarded after its document states are copied into the
ordinary FFN columns.

This remains a sufficiency screen, not an LM-training or production result.

## Frozen writer

- Checkpoint:
  `results/hotpot-semantic-address-t12-shared-base.pt`, SHA-256
  `a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b`.
- Schema `hotpot-semantic-address-t12-shared-base-v1`, model seed 8,209,
  20,000 from-zero steps.
- The checkpoint was sealed before T12's label-bearing probe.  It was trained
  only on protected natural text and the label-blind 2,405-document corpus.

## Frozen payload

For document `i`, tokenize `title + "\n" + text`, take the first 128 tokens,
run the frozen writer in BF16, mean-pool final normalized hidden states over
real (non-padding) positions, take coordinates 0 through 219, and L2-normalize.
The payload is then quantized through BF16.

No pooling, layer, coordinate, context, or normalization sweep is allowed.
All payloads must be finite/nonzero, with minimum BF16-to-FP32 cosine at least
0.9999.

## Fixed question/readout screen

Use exactly the raw-title matching, title-span removal, symmetric interaction
features, training-only standardization, float64 logistic reader, L2
coefficient 0.01, LBFGS limit 200, and threshold 0.5 frozen in T19b.

The question vector is produced by the same writer: after removing the two
matched title spans, tokenize the remaining property text, run up to 128
tokens in BF16, and take the final real token's normalized coordinates 0
through 219.  The writer never sees the yes/no label.

The logistic reader fits the existing 146 development-training labels and is
evaluated once on T12's 104 development questions.

## Physical contract

Reuse T19b's exact standard-Transformer layout:

- 132,800 title-code cells;
- 76,960 FFN gate-key cells;
- 2,405 gate thresholds;
- 2,405 up constants;
- 529,100 self-latent down-payload cells;
- total 743,670 writes, at most the 743,734 cap.

Candidate and untouched writer/control must retain identical class,
state-dict keys/shapes, parameter count, precision contract, and serving graph.

## Gates

All gates are required:

1. Every frozen hash/checkpoint metadata field and raw-only compiler interface
   is exact.
2. All 2,405 document payloads and 250 question vectors are finite/nonzero;
   minimum payload BF16 cosine is at least 0.9999.
3. Exactly two raw titles are matched in every question without support
   annotations.
4. The write count is exactly 743,670 and the exported model structure is
   unchanged.
5. Evaluation accuracy is at least 75%, at least ten points above T12's
   60.5769%, and at least ten points above T19b's 60.5769%.
6. Train/evaluation both contain both labels; the one fixed reader fit is
   finite with no retry.

Failure closes mean-pooled final self-latents as the payload.  Do not rescue it
with another layer, pooling rule, projection, context length, classifier, or
additional training.  Passing admits a separately preregistered from-zero
alternating write/read experiment with matched dense controls.
