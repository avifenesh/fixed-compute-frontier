# T32 information oracle — decoded-text invariant erratum

Status: **FROZEN AFTER CPU PREFLIGHT, BEFORE MODEL LOAD OR LABEL ACCESS**  
Date: 2026-07-31

## The failed assertion

The CPU preflight stopped on document `Dainan` because the frozen harness
required packing-tokenizer re-encoding to recover the identical stored ID
sequence. The 128 stored IDs decode to a string whose canonical re-encoding
has 127 IDs. No model was loaded, no answer value was accessed, and no score
exists.

The complete raw-corpus diagnostic found two noncanonical ID round trips among
2,405 records (`Dainan` and `Ioe no Iratsume`). In both cases, decoding the
canonical re-encoding produces the identical text. Zero decoded strings
changed.

## Why the assertion is not the information invariant

Let `P` be the frozen packer, `U` the frozen unpacker, `D` the tokenizer
decoder, and `E` its canonical encoder. Stage 0 proved

\[
U(P(x))=x
\]

for every admitted record. Function congruence immediately gives

\[
D(U(P(x)))=D(x).
\]

That is exactly the oracle interface: the strong reader receives `D(x)`.
Requiring `E(D(x))=x` additionally assumes that `D` is injective over token-ID
sequences. Byte-level BPE tokenizations need not be unique, so that stronger
identity is false even when the decoded text is bit-identical. It is not
needed by the physical candidate either, which consumes the recovered token
IDs directly.

## Sole admitted correction

Replace every `E(D(x))=x` check with the necessary invariant
`D(U(P(x)))=D(x)`. Keep the exact ID-level `U(P(x))=x` check. For selected
local windows, decode the exact selected recovered IDs without requiring
canonical re-encoding.

No record length, record content, window length, selector, shuffle, alias,
prompt, reader, continuation, scoring rule, threshold, or label boundary
changes. This correction is algebraically fixed before any model score and
cannot adapt to downstream accuracy. The next CPU preflight must traverse all
records, routes, and windows successfully before the single H100 scoring run.
