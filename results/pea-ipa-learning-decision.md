# PEA / IPA learned-screen decision

## Decision

Close hard chunk routing. Neither staggered PEA nor staggered IPA satisfies the
frozen admission gates. Do not spend a language-model or kernel run on either
architecture.

Retain one narrower result: a learned nonpairwise router can be causally useful.
The successor must transfer evidence between related tokens without hard chunk
boundaries and without adding QKV/FFN matrix multiplications, KV width, or a
second attention read.

## Formal result

All six global validity gates passed. Five seeds used equal initialization,
parameter counts, training data, and training order across the six arms.

| Arm | anchor same | anchor boundary | exact copy | distributed | overall |
|---|---:|---:|---:|---:|---:|
| ordinary | 63.223% | 64.282% | 82.461% | 58.135% | 67.025% |
| temperature 2 | 63.315% | 64.307% | 95.542% | 55.918% | 69.771% |
| PEA staggered | 72.427% | 68.057% | 95.415% | 56.807% | 73.176% |
| IPA staggered | 67.407% | 65.703% | 89.072% | 57.856% | 70.010% |
| PEA same offset | 72.319% | 67.817% | 95.664% | 56.240% | 73.010% |
| IPA same offset | 67.715% | 65.820% | 89.170% | 58.145% | 70.212% |

PEA improved overall accuracy by 6.151 percentage points over ordinary
attention and by 3.406 points over the temperature control. Its frozen `p1`
ablation reduced anchor accuracy by 5.688-8.081 points across every seed, with
a 6.533-point mean drop. The router was therefore learned and used, rather
than an inert mathematical decoration.

The architecture still failed:

- PEA's anchor mean exceeded IPA by only 3.687 points, below the frozen
  5-point requirement.
- Staggering improved PEA boundary accuracy by only 0.239 points over the
  same-offset version, below the frozen 5-point requirement.
- PEA's origin-8 attention distortion was 28.6%-32.0% relative to ordinary
  attention.
- PEA lost 1.328 points on the distributed task relative to ordinary. Most of
  its gain came from exact copy and the same-chunk anchor task, not a general
  contextual-reasoning improvement.
- IPA failed its overall, causal-use, and boundary gates.

## Consequence

The positive primitive is **evidence transfer**, not chunking: the score of a
token may profitably influence the retrieval weight of a related token. The
negative primitive is **hard partitioning**: a fixed chunk creates privileged
neighbors and discontinuous edges that staggering did not repair.

The next branch must use a boundary-free local relation, retain ordinary heads
as a protected path, and compare against a matched score-temperature control.
Its first gate must isolate relational retrieval from generic score sharpening.

## Frozen artifacts

- preregistration: `f8a217208d2bbaf1f06ae6fa4922d1786f48ed486d85e87553fefa41c13c621a`
- source: `117e058643483f93ef17952301ab1a5f0f715d2a9eefb73817ce246b161bcec8`
- integrity manifest: `2109b98cb68fda031203b2344c63699dafffcce26e3c3c98cae996e644858fd0`
- formal result: `98de3698cd506f61cf4a41b2c68313d1d3af9c6874007506c812dd45021bbe53`
- formal log: `fde934a252066a120d56fd9463198c4ad1a17938b8b46d1912e8f5b26da6c309`

The independent post-run audit reconstructed the reported deltas and all
selection decisions from the frozen result and returned `VALID`.
