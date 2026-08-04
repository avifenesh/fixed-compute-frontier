# T84 source audit v1 A

Bound source manifest SHA-256:
`efc659106c0d49d9792b547783733440f4dd3a3d4c0c04d60559cb8e8579577a`

Method: static source and manifest inspection only. No import, compilation,
test, fixture generation, timing, model execution, GPU use, or rental.

Verified in v1:

- frozen 106-statement family and simultaneous critical value;
- exact 20 percent relative boundary, one percentage point absolute boundary,
  five percent control floor, paired 16-seed construction, and gate signs;
- architecture parameter counts, lesions, recursive route shuffles, and the
  Tropical control equations.

Blocking findings:

1. The charged timer did not cover imports/setup or the final artifact
   scan/hash/fsync boundary.
2. Failed or killed attempts were replayable and had no durable opening marker.
3. Later receipts did not bind the complete prior artifact and decoder chain;
   raw timing files were not transitively bound.
4. Worker launch resolved through the base interpreter and depended on an
   unfrozen import path; bytecode writes outside the run root were possible.
5. Worker lifetime, CPU diagnostics, maximum VmHWM, and decoder high-water
   accounting were incomplete.
6. Decoder tests did not enumerate all 32 boolean inputs.

Verdict: **REVISE**. This manifest authorizes no execution.
