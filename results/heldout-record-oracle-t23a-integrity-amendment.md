# Held-out forced-read record oracle T23a — integrity amendment

Status: **FROZEN AFTER THE INVALID FIRST RUN, BEFORE ANY T23a RERUN**  
Date: 2026-07-31

The first T23a execution completed but classified itself as
`invalid_experiment`.  Its result and log SHA-256 values are respectively
`e5fe78e44c0ed5c6737acb6258d067e1460d11fd41a2beb8db280b60b0d27536`
and
`716bd4b3bbe9568b5de68d8107e87012616c8372779c4b130a0edf9019ec76f2`.

The only failed mandatory integrity gate was
`q16_levels_and_bf16_exact`.  The frozen preregistration requires that Q16
records use the 16 frozen level **indices** and that those indices survive a
BF16 round trip.  The implementation instead compared floating-point level
values with `torch.equal`, which is a stricter and representation-dependent
condition not stated by the preregistration.

The rerun changes only that integrity implementation:

1. decode every Q16, projected-Q16, and matched-writer record to its frozen
   integer level index;
2. require all decoded indices to be in `[0, 15]`;
3. require the decoded index tensors to remain byte-identical after a BF16
   round trip;
4. report each component separately.

No seed, input, split, target, record value, forward pass, gradient, optimizer,
schedule, update count, control, evaluation, threshold, classification rule,
or model changes.  The invalid first result is retained.  The corrected rerun
must start from zero and is authoritative only if every original integrity
gate passes.
