# Gauge-funded one-curvature energy screen — decision

Status: **formal package rejected; one-curvature capability retained**  
Date: 2026-07-27

## Frozen result

The sealed teacher-free energy screen returned `energy_screen_pass=false`:
three of five formal worlds passed. The result file is
[`gauge-embedded-curved-keys-energy.json`](gauge-embedded-curved-keys-energy.json),
SHA-256 `a70b6aa473f59261600f1720e4e48c356542764d23a7a6089d6bc78061d6fd66`.

The only failed gate in either failed world was the auxiliary claim that
centered scale-funded G1 would have eight-record OOD NLL no more than 0.10
above matched centered rotation-funded G1:

- seed 89 failed;
- seed 167 failed;
- seeds 127, 211, and 257 passed.

Therefore the claim that the scale chart is consistently at least as well
conditioned as the rotation chart is rejected. The frozen H100-kernel and
language-model admissions remain false.

## Result that survived inside the rejected package

The uncentered scale-funded G1 primary passed every direct capability,
invariance, OOD, ablation, parameter, bootstrap, and paired-restart gate in all
five worlds. This is a retained finding, not a retroactive pass of the package.

Across the five worlds:

- mean primary accuracy: `0.9977783203125`;
- mean bilinear accuracy: `0.299169921875`;
- worst paired-bootstrap 95% lower accuracy advantage: `+0.6571014404296875`;
- worst paired-bootstrap 95% lower NLL advantage: `+1.3727696339918902` nats;
- primary accuracy after eight-record Gaussian shift: `0.9762` to `0.9944`;
- primary accuracy after eight-record Laplace shift: `0.9825` to `0.9948`;
- all 24 position permutations, independent coordinate-sign flips, quadratic
  derangement, curvature ablation, exact parameter/byte accounting, and
  finiteness gates passed in all worlds.

The task has no GECK teacher: it directly asks an attention head to retrieve
the record with the largest requested squared content coordinate. Ordinary,
gauge-linear, and position-only arms remain near their linear ceiling. Moving
the complete quadratic feature block to a different record destroys the
candidate result, and zeroing its sole curvature returns it near the linear
control.

## Retained algebra

For each norm-free RoPE pair, an existing full scale gauge can expose

`v' = v + tanh(s) * u^2`.

At `s=0` this is exactly the ordinary bilinear head. It adds no learned scalar,
dense projection, or KV-cache coordinate. Its token epilogue is one square,
one coefficient multiply, and one add per pair. Coefficient extraction from
the existing pivot weights is not free and must be charged by any kernel gate.

Four pairs made a single curvature direction sufficient; G2 gave no capability
gain on this task despite twice the nonlinear scalar work. The useful retained
primitive is therefore **one gauge-funded quadratic content feature per RoPE
pair**, not bi-curvature G2 and not a privileged scale chart.

## Next admission

Do not repair the failed centered scale-vs-rotation claim. A successor must use
new unseen worlds and state the narrower hypothesis:

1. gauge-funded G1 improves a teacher-free content-addressing task;
2. the chart/orientation is a parameterization choice, not a claimed source of
   extra expressivity;
3. an ordinary signed-linear retrieval task must show no-harm under matched
   training;
4. only a pass admits coefficient-access and fused-epilogue benchmarking.

No language-model claim is admitted until the narrower gate, physical H100
gate, and a matched LM screen all pass.
