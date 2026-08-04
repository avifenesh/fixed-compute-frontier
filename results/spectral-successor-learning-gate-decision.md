# Spectral successor learning gate: closed negative result

Date: 2026-07-30

## Decision

**CLOSED. Do not tune or advance this formulation.**

The run was stopped as soon as a preregistered gate became irreversibly false.
This is intentional sequential falsification, not a missing result.

## Fatal observation

On structured seed 731, both NTP and spectral successor first exceeded the
90% first-two-plan-token threshold at the 320-step checkpoint:

| arm | step 160 | step 320 | charged work at crossing |
|---|---:|---:|---:|
| NTP | 62.89% | 99.22% | 7,430,461,194,240 |
| spectral successor | 37.11% | 100.00% | 7,515,458,764,800 |

The spectral arm costs 1.1439% more work per step. Its measured
compute-equivalent acquisition gain is therefore

`7,430,461,194,240 / 7,515,458,764,800 = 0.98869x`.

That fails the T1 admission threshold of 1.25x and is not remotely close to
the project threshold of 1.5x. Since the rule requires both seeds to pass,
later measurements cannot reverse the failure.

## Stronger diagnostic

At step 160:

- NTP: 62.89% first-two exact;
- spectral: 37.11%;
- scrambled spectral: 28.91%;
- compressed MTP: 30.47%;
- bag-of-future: 19.92%.

The ordered spectral target carries some useful task-specific signal relative
to the other auxiliary targets, but the ordinary NTP gradient learns the plan
substantially faster. By step 320 all arms solve the task. The candidate adds
conditioning pressure; it does not expose a missing capability or eliminate a
large training cost.

The terminal plan NLL among completed structured arms also rejects a semantic
story: the scrambled spectral control reached a lower plan NLL than the true
spectral arm (0.000219 versus 0.000565 at step 800). Any late improvement is
therefore compatible with generic auxiliary regularization.

## What is retained

1. The spectral code is an ordered, fixed-width suffix representation and the
   Stage 0 algebra is valid.
2. A deterministic future target adds no corpus information. In this test it
   did not improve the key credit path beyond NTP.
3. Future work should not search for another decorative auxiliary loss. To
   clear the project bar, a method must remove a large source of training work
   or change the scaling of feature acquisition.

Raw evidence is in `results/spectral-successor-learning-gate.log`. The full
matrix was not completed because the frozen decision was already false.
