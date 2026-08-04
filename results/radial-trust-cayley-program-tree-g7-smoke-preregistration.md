# Radial-trust Cayley program tree G7 smoke preregistration

Run exactly one optimizer step at the frozen language-pilot microbatch and
sequence length before committing to the 10M-token run.

Frozen artifacts:

- Module: `cf64ab47717ec0cb8a8720fbd768adf48ca9785f0169253966f5a1539ff70226`
- Pilot: `79cdccbbaffb11428f5fc5e1368c9f690a56538f74d7cafdb8c5914c7f806ca5`
- Smoke executable:
  `7e228e7b23703bea6105523c48c87adf547e61876fdbc54e9465ef0a4431af6f`
- Stage-0 result:
  `973a2a210dce047544347c7b2cecbca1805c4f707553bcc2753c47d3740cfa6e`

Protocol: seed 815, sequence 512, microbatch 8, AdamW with the frozen pilot
settings, route auxiliary 0.01, gradient clip 1.0, exact frozen training shard
at batch zero.

Pass requires artifact integrity plus finite language loss, finite router
auxiliary, finite pre-clip gradient norm, successful optimizer step, and peak
reserved memory below device capacity.  No quality claim is made.  Failure
closes the exact candidate before the language pilot.
