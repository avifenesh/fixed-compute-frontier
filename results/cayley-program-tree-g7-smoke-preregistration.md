# Cayley program tree — G7 execution smoke preregistration

Status: **frozen before first GPU execution of the program tree**  
Date: 2026-07-30

## Purpose

This is a feasibility check, not a capability result.  Run one optimizer step
of the exact preregistered language candidate at sequence length 512 and
microbatch eight on the dedicated G7 instance.  Record loss, gradient norm,
elapsed time, peak allocated and reserved VRAM, software versions, parameter
ledger, source hashes, and data hash.

## Gate

The unmodified candidate must complete a forward pass, backward pass, clipped
gradient, and fused AdamW step without OOM or nonfinite values.  Peak reserved
VRAM must fit the physical device.  Failure permits only semantics-preserving
training-memory engineering such as activation recomputation; it does not
permit a smaller batch, altered model, altered initialization, or changed
language-pilot claim without a new preregistration.

Wall-clock speed from this Python/scatter reference is diagnostic only.  No
serving-efficiency claim can use it.
