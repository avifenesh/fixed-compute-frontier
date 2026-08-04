# T32R capability audit — routing provenance erratum

Status: **CORRECTED BEFORE ANY H100 IMPLEMENTATION OR TIMING**  
Date: 2026-07-31

The first capability audit described T32R's document handles as supplied or
privileged.  Source inspection shows that wording was wrong.

The frozen router:

1. reads only the raw question string;
2. case-folds it and the 2,405 corpus titles;
3. finds literal title occurrences;
4. selects two longest non-overlapping matches;
5. replaces those surface spans with input-only handles.

The evaluator's `supporting_titles` field is validated as part of the sealed
schema but is not read by routing or prompt construction.  It may be consulted
after routing only to measure route correctness.

The source-provenance correction is therefore:

> T32R has a deterministic raw-only exact-title candidate generator.  It does
> not read evaluator route labels.

A subsequently preregistered CPU correctness audit narrowed this further.  The
longest-two matcher agrees with sealed supporting-title pairs on 102/104
questions, not 104/104.  Third title mentions in predicates/categories can
displace one compared argument.  Therefore autonomous route **provenance**
passes while autonomous route **correctness** fails.  See
[`t32r-exact-title-router-stage0-decision.md`](t32r-exact-title-router-stage0-decision.md).

Neither correction re-admits the frozen H100 implementation.  The direct
`uint16` representation still byte-dominates the BF16 radix table, and the
rank-32 natural bottleneck remains unvalidated.

## Evidence

- information-oracle source SHA-256:
  `58431109b75675d145c76cd9d953e5ab7a5f4e6c0d42a01093c673de5fa5286f`
- handle-census source SHA-256:
  `dbab287ff5dd98136aeba6c538e8336239f5da0891a45718fd8b51654034cc5a`
- corrected capability-audit SHA-256:
  `c00078fe7b852dad4cad14c816eface1a3b9b920dc24543e8afe1526cbc4f0cd`
