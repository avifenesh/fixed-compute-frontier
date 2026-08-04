# Self-indexed sparse latent — natural-locality fatal gate

Status: **frozen before model activation capture**  
Date: 2026-07-30  
Hardware: one dedicated AWS `g7e.4xlarge`, RTX PRO Server 6000 96 GiB

## Question

Can a fixed-degree graph propagate the active feature addresses of a real
language model without recomputing a dense global router at every layer?

This is the fatal empirical assumption behind the self-indexed sparse latent.
The gate does not train the proposed architecture.  It asks whether ordinary
language representations contain enough local, width-stable transition
structure to justify that much more expensive experiment.

## Frozen models and data

One family at three widths is used so width trends are interpretable:

| model | revision |
|---|---|
| `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` |
| `HuggingFaceTB/SmolLM2-1.7B` | `effd688a12921b4cc83e3312b6feb579f70f9c71` |

Text is `Salesforce/wikitext` revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, configuration
`wikitext-103-raw-v1`.  Graph edges are fit on the first 8,192 usable training
tokens.  All decisions use the first 4,096 usable validation tokens, never
used for edge selection.  Sequence length is 256.

## Feature events

For each selected MLP, capture the exact tensor entering `down_proj`:

`SiLU(gate_proj(x)) * up_proj(x)`.

The 64 largest-magnitude coordinates per token become events.  Sign is part of
the address, giving `2M` possible addresses for intermediate width `M`.
Selected layer pairs are the adjacent pairs beginning nearest 25%, 50%, and
75% of model depth.

The event budget is deliberately fixed across all widths:

- active events `k = 64`;
- outgoing degree `d = 16`;
- at most `k*d = 1,024` edge records read and emitted per transition.

## Edge compiler and predictions

For every source address `i`, count co-activation with destination address `j`
on training tokens.  Retain the 16 destinations with largest conditional
frequency and their conditional probabilities.  At validation time, every
active source emits its 16 stored destinations weighted by source magnitude
and conditional probability.  Equal destinations are summed; the 64 largest
scores are the predicted next events.

Controls:

1. **global frequency:** always predict the 64 most frequent training
   destination addresses;
2. **shuffled source rows:** preserve the validation source-event marginal but
   give each token another token's active source set before graph propagation;
3. **top-k energy:** independently measure whether 64 events contain enough of
   the actual full MLP-intermediate squared energy.

## All-or-nothing survival gate

Every layer pair at every width must satisfy all of:

1. median top-64 squared-energy fraction at the destination is at least 0.90;
2. graph recall@64 of the actual destination top-64 addresses is at least 0.80;
3. graph recall exceeds global-frequency recall by at least 0.20 absolute;
4. graph recall exceeds shuffled-source recall by at least 0.20 absolute;
5. source-conditioned graph quality does not trend downward with model width
   at all three depth fractions.

These are large structural thresholds, not sub-percent acceptance rules.
Failure means that fixed-degree address propagation is not supported by
ordinary language features and no sparse-graph model or GPU kernel is trained.

Passing is still not a breakthrough.  It would admit a from-scratch trainable
addressing experiment with dense, PEER-style retrieval, and matched table/RAM
controls.  It would not establish language-model quality, novelty, or GPU
efficiency by itself.

