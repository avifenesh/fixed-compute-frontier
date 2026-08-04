# Joint in-place weight compiler T19 — theory audit

Status: **original rank gate rejected; narrower algorithmic gate admitted**  
Date: 2026-07-31

## Correction

The T19 hypothesis proposed showing that compiler writes store more independent
facts than the rank of a matched gradient update.  That is not a valid
separation.  A single minibatch gradient can be low rank, but successive
gradients can span the entire writable parameter subset.  Comparing against a
single-update rank would make the control artificially weak.

There is a second limit that must be explicit.  If candidate and control export
the same Transformer class, shapes, precision, and number of parameter bits,
then they have the same representable function class.  A training-only
compiler cannot create asymptotic serving capacity.  Its possible gain is an
algorithmic one: reach a useful, interference-resistant organization of those
weights using less charged training work than frontier gradient training.

## Rejected naive writer

For a routed record with one-hot address `e_i`, directly assigning a value
column `D e_i = z_i` is equivalent to one gradient step of size one on

`0.5 * ||D e_i - z_i||^2`

when that column is initialized at zero.  Therefore direct value assignment by
itself is not a research advantage.  It is only useful if the complete method
also solves a hard shared problem such as discrete addressing, global
allocation, collision control, or symbol regeneration, and beats a control
given the same derived targets.

## Surviving question

T12 already showed that a raw-title token superposition gives collision-free
32-dimensional addresses for all 2,405 real document titles and all 208
development question surfaces.  T15–T17 showed that an ordinary LM does not
spontaneously expose such an exact address after training.

T19a therefore moves the code into ordinary token-embedding cells at the start
of training and co-trains the reader from zero.  The compiler performs no
semantic inference: each tokenizer row receives a deterministic bipolar code,
and each document address is the normalized mean of its title-token codes.
The candidate and controls receive exactly the same raw-only title-to-code
objective.  The only difference is direct deterministic initialization and
preservation of the code cells.

This tests an algorithmic separation rather than a function-class claim.  If a
twice-trained dense control matches the candidate, or if shuffling the code
cells does not remove the candidate's advantage, this T19a route closes.

## Consequence for the full goal

Passing T19a would establish only the missing exact real-title address inside a
standard from-zero Transformer.  It would admit, not prove, a subsequent
raw-prose fact extractor and digital payload writer.  The production goal
still requires a fresh document-disjoint knowledge and reasoning gain at
identical serving cost.
