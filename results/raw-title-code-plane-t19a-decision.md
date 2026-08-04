# Raw-title code plane T19a — decision

Status: **FAIL; close deterministic code initialization as the advantage**  
Date: 2026-07-31

## Decision

T19a does not pass its exactness or strict-separation gates.  Do not rescue it
with more code dimensions, wrappers, steps, learned projections, or relaxed
thresholds.

The retained result is narrower: co-training from zero solves the real
multi-token title-address interface almost completely, and deterministic code
writes are causal, but the same explicit routing targets let ordinary gradient
training approach the result.  Address initialization is therefore not the
source of a large fixed-compute capability gain.

## Sealed result

| arm | canonical | held wrappers | natural question surfaces |
|---|---:|---:|---:|
| compiled 1x | 100.0000% | 99.6674% | 99.5192% |
| gradient 1x | 100.0000% | 97.3389% | 98.5577% |
| gradient 2x | 99.9584% | 97.0686% | 98.0769% |

The candidate advantage over the twice-trained control was only:

- +0.0416 points canonical;
- +2.5988 points on held wrappers;
- +1.4423 points on natural question surfaces.

The preregistered requirement was at least +10 points on every surface and
100% candidate accuracy with positive margins.  The candidate made 16 held
wrapper errors and one natural-surface error, so both primary gates failed.

## What did pass

- Only 132,800 embedding entries were written, 17.86% of the 743,734-entry
  budget.
- Compiler cells remained bit-exact through 3,000 mixed steps.
- Candidate and controls had the identical 36,577,152-parameter ordinary
  Transformer structure.
- Candidate natural NLL was 5.69105 versus 5.70936 for the matched 1x control,
  a 0.3207% improvement.
- Shuffling the code cells reduced natural-surface routing from 99.5192% to
  59.6154%, a 39.9038-point causal drop, and the state restored exactly.
- All training was finite with zero failed/retried steps.

The code is real and used; it simply does not create a qualitative advantage.

## Interpretation

The compiler provided an early optimization lead, but gradient training erased
most of it.  This confirms the T19 theory audit: when both arms receive an
explicit title-to-code objective, deterministic embedding writes are a warm
start inside the same function class, not denser knowledge acquisition.

The address problem should no longer carry the research claim.  Future payload
experiments may give every arm the same deterministic/co-trained address and
must isolate the claimed advantage in raw-prose retention or structured
payload acquisition.  The proven digital relation-plane separation remains
the relevant positive result; title-code initialization does not add another
one.

## Evidence

- Result SHA-256:
  `1b324ac013ccd4bd4dc57d18fae180d73dccb66bf151edee30bb92bec695e51a`.
- Source SHA-256:
  `73585ed661c6507f3a30887036d9346146f2bc4d6f47ecd865e46421460ce366`.
- Preregistration SHA-256:
  `5b6795980ae80d8227cd22b2cee58915340f4042888fe3f825e3f4e1e64ca715`.
- H100 training times: 205.02 seconds compiled 1x, 206.55 seconds gradient
  1x, and 413.95 seconds gradient 2x.
