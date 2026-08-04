# T84 product-semiring attention — independent audit A

Date: 2026-08-02  
Scope: algebra, prior-art/absorption risk, CPU witness validity, full runtime cost, and the claimed counterexample to arXiv:2601.09775.

## Bottom line

The scalar algebra, the revised Bellman proposition, and the primary within-head shared-score construction are coherent, and the counterexample to *The Geometry of Thought* is decisive. The proposed model, however, is not yet a causally diagnostic CPU experiment. Its two optimization domains substantially reproduce the inductive bias and task families already occupied by *Tropical Attention*, while its fuzzy-retrieval domain can reward trivial task-family routing unless the named mixed compositions are made operationally inseparable.

The residual research seam is real but narrow: preserve an ordinary language-capable softmax path while using the **same learned score matrix** for a per-channel max-plus path, then beat faithful Tropical Attention and strong narrow-head/cross-token-max controls at complete cost. The screen should be revised around that seam before CPU admission.

## 1. Algebra audit

### Equations 1–3: pass, with domain qualifications

For \(\beta>0\), finite legal inputs, and at least one unmasked source,

\[
B_\beta=\beta^{-1}\log\sum_j e^{\beta(s_j+u_j)},\qquad
Z_\beta=\beta^{-1}\log\sum_j e^{\beta s_j},\qquad
A_\beta=B_\beta-Z_\beta
\]

are well-defined. At \(\beta=1\), exponentiating the difference gives

\[
e^{A_1}=\frac{\sum_j e^{s_j+u_j}}{\sum_j e^{s_j}}
=\sum_j\operatorname{softmax}_j(s)e^{u_j},
\]

so Equation 3 is exact. It is specifically positive-valued attention in log coordinates, not ordinary signed-value attention. The limits \(B_\infty=\max_j(s_j+u_j)\) and \(A_\infty=B_\infty-\max_j s_j\) are also correct. The text should state \(\beta>0\), define \(n\) as the number of legal/unmasked sources, and exclude all-masked rows; otherwise subtraction can encounter an undefined \((-\infty)-(-\infty)\).

There is a naming mismatch to repair. A Cartesian product of the log and max-plus semirings is indeed a semiring, but the proposed Euclidean branch uses ordinary signed softmax values, not log-positive coordinates. Moreover, softmax normalization subtracts \(Z_\beta\), whereas raw semiring matrix multiplication produces \(B_\beta\). The implemented block is therefore a heterogeneous concatenation/direct sum unless the log branch is actually represented in the log semiring. “Product-semiring” is a useful design motivation, but not yet a literal algebraic description of the whole deployed operator.

### T84.1: pass

Writing \(\operatorname{LSE}_\beta(x)=\max x+\delta_x\) gives \(0\leq\delta_x\leq\log n/\beta\). Thus

\[
0\leq B_\beta-B_\infty\leq\log n/\beta,
\qquad
A_\beta-A_\infty=\delta_{s+u}-\delta_s,
\]

and the stated absolute bound follows. The uniform bound is essentially tight: take \(s\) tied across all \(n\) entries and choose \(u\) so one entry dominates; the error approaches \(\log n/\beta\). Hence logarithmic growth of \(\beta\) is necessary for a *uniform worst-case* fixed-error guarantee, not for every input distribution.

### T84.2: pass in its current, repaired form

The raw update \(Y_{ic}=\max_j(s_{ij}+u_{jc})\) is exactly max-plus matrix multiplication. With a fixed \(S\) and no interposed residual, projection, normalization, or nonlinearity, induction gives maximum scores over length-\(L\) walks. The current draft correctly says that the normalized \(A_\infty\) changes multi-step path values and assigns the Bellman claim only to unnormalized \(B_\infty\).

That scope must survive implementation. Recomputing \(S\) from each layer's hidden state, normalizing the tropical state between steps, or applying an output projection destroys the literal fixed-graph path theorem. Those operations may still learn a useful network, but cannot be cited as exact Bellman execution.

## 2. Is the edge absorbed by existing mechanisms?

### Ordinary multi-head attention: not an exact matched-cost refutation, but a strong absorber

The local distinction is real. A standard head has one source distribution for all value channels, while \(\max_j(s_{ij}+u_{jc})\) may select a different \(j\) for every channel \(c\). Many one-dimensional heads can recover channel-specific selections, and a head can place a channel-specific \(u_{jc}\) into its logits (for example via a constant query coordinate). Retaining the same full-rank shared relation score for every such channel, however, duplicates score capacity; at fixed total Q/K width, narrow heads reduce each matcher's rank. Therefore T84 has a plausible parameter-allocation/inductive-bias advantage, not a distribution-free expressivity separation.

The mandatory comparison is not merely “more heads.” It must match total Q/K/V/O parameters, active dimensions, depth, training/tuning budget, and complete runtime, and it must include one-dimensional heads plus a version allowed to construct relation-conditioned logits. A loss only against the baseline head count says little.

### Maxout/MLP: the listed control is underspecified

[Maxout](https://proceedings.mlr.press/v28/goodfellow13.html) takes maxima over affine responses, but a tokenwise maxout/MLP does not itself aggregate relation-conditioned values across source tokens. It can absorb the edge only after some cross-token communication or with additional depth. The faithful control is a score-conditioned cross-token max/maxout aggregator, or the narrow-head construction above, not simply an equal-parameter tokenwise MLP.

### Tropical Attention: absorbs the broad mechanism and most proposed evidence

The direct collision is substantial. [Tropical Attention](https://arxiv.org/abs/2505.17190) already defines tropical Q/K/V projections, a tropical Hilbert score, and the exact channelwise aggregation

\[
C_i=\max_j\{S_{ij}+v_j\}.
\]

It reports length/value OOD results on Floyd–Warshall, knapsack, subset sum, balanced partition, and related combinatorial tasks, and proves max-plus DP simulation. It explicitly leaves generative/autoregressive scaling and tropical runtime/memory overhead open, and names hybrid semiring architectures as future work. Thus the max-plus primitive, Bellman interpretation, and synthetic DP-OOD advantage are occupied. A current primary-source search did not find a paper that already establishes T84's exact shared-score softmax/max-plus feature partition, but that is a search result, not a novelty proof.

The surviving edge is consequently **hybrid coexistence plus genuinely shared relation scores in a language-capable setting**. The revised primary construction now tests that exact object: it computes one \(S\) per head and partitions value channels. Whole-head allocation is correctly demoted to an ablation/fallback because it tests head heterogeneity rather than a product over one relation graph. A shared-versus-independent-score ablation remains necessary to attribute any gain specifically to relation sharing.

## 3. CPU witness and the 20% claim

The statistical gate is conservative, but statistics cannot repair a non-diagnostic intervention. As written, the witness cannot causally support a major-stage \(\geq20\%\) improvement:

1. Graph/Viterbi and knapsack/segmentation are generated by the operator being supplied and overlap the strongest Tropical Attention evidence. A gain there establishes operator alignment, not a broader reasoning capability.
2. Interleaving separate DP and fuzzy-retrieval families permits domain classification and static head routing. The retrieval family mainly demonstrates that retaining softmax avoids an all-tropical regression.
3. “Mixed compositions” and “task-separable controls” are named but not operationalized. Without a single example requiring both reductions over the same latent relation graph, the central product claim is not identified.
4. “All-tropical” is not automatically a faithful Tropical Attention control. T84's dot-product score plus max-plus value reduction differs from that paper's valuation map, tropical projections, Hilbert metric, initialization, and devaluation.

Revise the witness before running:

- Keep the revised primary object fixed to feature-group sharing of one \(S\), and add a \(2\times2\) ablation over `{softmax, max-plus}` aggregation and `{shared, independent}` score graphs. Make branch normalization identical or identity-gated so the all-softmax model is truly the baseline; otherwise the claimed exact baseline containment is false.
- Construct within-instance compositions in which the same edges support (a) an averaging/signed-evidence subanswer and (b) a best-path subanswer, with a final answer requiring both. Pre-register branch-lesion and score-unsharing tests that selectively break the corresponding subanswer.
- Include task-label, length, vocabulary, and surface-statistic probes showing that a family router cannot solve the suite. Hold out operator compositions, not only lengths and values.
- Reproduce faithful Tropical Attention and the strongest narrow-head/recurrent controls under the same total budget. Replace generic “maxout/MLP” with a cross-token score-conditioned max control.
- Treat the first CPU stage as an operator-learning falsifier. Even a 20% simultaneous lower bound on operator-generated tasks advances the mechanism only to a non-synthetic or language-reasoning test; it is not itself evidence for a 20% complete model-level gain.

With these revisions, a CPU run can decide whether shared mixed algebra adds anything beyond routing and prior tropical attention. It still cannot adjudicate production cost.

## 4. Full GPU/runtime risks

Parameter count, KV width, and \(O(n^2d)\) arithmetic do not imply runtime parity. [FlashAttention](https://arxiv.org/abs/2205.14135) gains speed by tiling and fusing the score, online softmax, and weighted-value reduction to minimize HBM traffic. A tropical kernel can also tile, but \(\max_j(S_{ij}+U_{jc})\) needs an independent accumulator and winner for every query-channel pair. Likely costs include:

- broadcast/add/max reductions that do not map to the tensor-core GEMM used by \(P@V\), with register/shared-memory pressure and lower occupancy;
- argmax indices stored or recomputed for every query and value channel in backward, plus sparse winner-only gradients and tie-sensitive numerics;
- dead or monopolizing routes during training. [Semiring Activation](https://arxiv.org/abs/2405.18805) documents that standard Xavier/Kaiming assumptions do not transfer and motivates fair tropical initialization because non-winners receive zero gradient;
- signed-residual-to-finite-log-potential and inverse mappings, currently unspecified. Log/ReLU/exp-style mappings introduce zeros, \(-\infty\), overflow/underflow, precision, clipping, and semantic distortion risks;
- causal-mask sentinel safety, mixed-precision reproducibility, branch-specific launches, concatenation/normalization overhead, and loss of existing fused-attention paths;
- prefill and decode behaving differently: equal KV bytes do not make the per-token reduction equal, and cached transformed potentials may require extra precision or metadata.

The complete ledger must include forward and backward wall time, peak workspace, winner storage/recompute, compilation/autotuning and failed kernels, branch fusion, potential transforms, prefill/decode p50 and p95, and energy if capability-per-cost is used. Compare a fused tropical kernel against the platform's production SDPA/FlashAttention path, not a naive PyTorch attention baseline. The published Tropical Attention paper itself flags nontrivial tropical runtime and memory overhead and has not established autoregressive language scaling.

## 5. arXiv:2601.09775 Theorem 1

The counterexample is correct. [Theorem 1](https://arxiv.org/html/2601.09775v1) claims

\[
\lim_{\beta\to\infty}\operatorname{softmax}(\beta A)V=A\otimes_{\max+}V.
\]

For \(A=[10,0]\) and \(V=[0,100]^\top\), the unique score maximizer is index 1, so the left side tends to \(0\). The tropical product is \(\max(10+0,0+100)=100\). Even the row-normalized tropical value is \(100-10=90\), not \(0\).

The proof's invalid step is the assertion that the score maximizer \(j^*=\arg\max_j A_{ij}\) also realizes \(\max_j(A_{ij}+v_j)\); subtracting the row maximum does not preserve the argmax after channel-dependent \(v_j\) is added. Ties create a second failure: softmax converges to the average of values over tied maximum-score indices, not to an arbitrary tropical winner. Therefore Theorem 1 and its Bellman/path corollary are false as written. This does not invalidate the genuine log-sum-exp-to-max limit used in T84.

## Decision

The math and revised shared-score architecture are strong enough to keep the lane alive, but the current CPU witness does not isolate that mechanism from supplied max-plus structure, task routing, or existing Tropical Attention, and it cannot settle production cost. Repair the algebraic naming/baseline-containment details and causal witness first.

REVISE

## Re-audit after revision

Date: 2026-08-02

The revision resolves the blockers that produced the initial `REVISE` decision:

- It correctly renames the deployed object as heterogeneous/direct-sum attention and confines the Cartesian-product-semiring statement to the log-domain reference construction. Equations 1--4 and T84.1/T84.2 remain correct under the now-explicit `beta>0`, finite-source, and nonempty-mask conditions.
- The primary operator is now frozen unambiguously: one within-head score matrix is shared by disjoint signed-softmax and real-valued max-plus value coordinates. Direct finite `W_V h` potentials remove the previously unspecified log/exp/sign conversion and most of its numerical/semantic risk.
- Identical group normalization is applied to the primary candidate and matched all-softmax control, the conventional block remains a separate control, and baseline containment is no longer overstated.
- The CPU ladder now makes both reductions necessary within every instance, progresses from supplied-score unit identity to a learned relation and then a permuted causal weighted automaton, forbids privileged DP structure, and includes branch, shortcut, serialization, gradient-health, and score-sharing kills. This is sufficient to test the claimed mechanism rather than merely classify task families.
- The fatal absorbers are represented: faithful published Tropical Attention, many narrow softmax heads, branch-specific scores, whole-head splitting, a score-conditioned cross-token max/maxout operator, recurrent models, finite/annealed-temperature variants, and both parameter- and measured-time matching.
- CPU success is explicitly reject-only. It cannot establish a 20% major capability result, production cost, or a smarter served model; it can only earn a subsequent frozen language-model integration. This removes the earlier causal overclaim.

The preregistration still must freeze the exact data generator, XOR balance/independence checks, thresholds, RMS axes and epsilon, control-stage ordering, seed/tuning budgets, time-matching rule, and simultaneous-comparison procedure. It must also keep T84.2's exact walk interpretation restricted to fixed `S`; success on the learned automaton would be empirical evidence, not proof that a normal Transformer stack literally executes that theorem. These are preregistration deliverables, not reasons to close or demand another conceptual revision.

No CPU or GPU run is admitted by this decision.

ADMIT PREREGISTRATION

## Adversarial audit of CPU preregistration v1

Date: 2026-08-02  
Object: `shared-score-heterogeneous-attention-t84-cpu-preregistration-v1.md`

### Verdict

The task concept is now capable of isolating the proposed mechanism: conditional on the shared score vector, the soft and max bits are independent and balanced; XOR makes either bit alone useless; and the model must recover both relations from the same mixed record set. The reject-only scope is also appropriate. The document is nevertheless not executable as a preregistration without material choices being made after acceptance. Several choices could change which architecture wins, and two current rules can directly create a false positive. This is a `v2` repair, not grounds to close T84.

### Required edits

#### 1. Freeze the actual sample and mask

1. **Remove OOD distribution leakage.** Learning-rate selection currently uses OOD-L16 while OOD-L16 is also a primary held-out slice. Tune only on the stated in-support development distribution, or introduce a calibration length that is not in the primary family. OOD-L16 must not be inspected during any model or hyperparameter choice if it is called held out.
2. **Define the final-query source set.** An ordinary non-strict causal mask lets the final query attend to itself, although the target aggregates records only. A strict causal mask gives the first record no source. Freeze one exact mask matrix and either add a BOS token, exclude final-query self by an explicit common mask, or include the query contribution in the target. State whether record-token outputs matter and use the identical matrix in every control and reference check.
3. **Specify the RNG contract, not just top-level seeds.** Freeze the slice-index ordering, PCG64/NumPy version, QR sign convention, draw order, rejection-balancing algorithm, permutation timing, and separate substreams for data, initialization, and shuffling. Define the seed map for every tuning/evaluation architecture and ensure paired architectures receive byte-identical training batches and evaluation records. Hash the mixed matrix and all fixed evaluation tensors before training.
4. **Make leakage kills implementable.** Freeze whether surface features are computed before or after orthogonal mixing, their exact concatenation and standardization, logistic-regression solver/penalty/regularization/max-iterations/seed, and the query-only and one-record-only MLP inputs, record choice, optimizer, LR selection, initialization, and evaluation aggregation. “A query-only or one-record-only MLP” is presently ambiguous between one and two controls.
5. Define the Stage-0 away-from-tie margin and what is included in the “bitwise artifact”; timestamps and unordered manifest metadata cannot be part of a bitwise replay test.

#### 2. Repair the control set and failure semantics

1. Add `soft_narrow_grouped`: one-dimensional heads followed by the same frozen groups of four-channel normalization. `soft_narrow` and `soft_grouped` separately do not test the strongest ordinary combination.
2. Add a T84-score **all-max** control. The official Tropical Attention control changes the valuation, score, projection, and inverse map, so it cannot establish that heterogeneous reducers are needed rather than merely max-plus aggregation. Also add or precisely identify the promised score-conditioned cross-token max/maxout control. If `hybrid_independent` is intended to fill that role, say so and explain why; it currently tests independent score allocation, not generic post-communication maxout.
3. State the exact post-attention normalization for `hybrid_independent`, `hybrid_whole_head`, `finite_beta_8`, and the all-max control. For fair attribution, the independent dimension-four heads should receive the same dimension-four non-affine normalization; the whole-head rule must say whether normalization is over four or eight channels.
4. Pin `tropical_official` to an immutable upstream commit and vendored-file SHA before implementation, plus its exact causal-mask adaptation, `tropical_norm`/`symmetric` settings, initialization, FFN width, and parameter count. The live [official repository](https://github.com/Baran-phys/Tropical-Attention) describes “2.0.0” on `main`, while its published release list is not an immutable v2.0.0 release; a moving branch and a source revision selected in the post-implementation manifest are not preregistration. Its native `TropicalLinear` initialization also differs from the backbone's Xavier rule and must be explicitly retained or replaced.
5. Either include the previously fatal learned-temperature/sparse attention and recurrent controls, or narrow v1's positive sentence to the exact frozen controls and state that these absorbers are deferred. No v1 result may say it beat “every” same-cost learned alternative.
6. **Reverse the crash rule for controls.** Assigning a crashed control its worst error makes the candidate look better. A candidate crash fails v1; a required-control semantic/numerical crash must make that contrast fail or invalidate the protocol. Only a proven common infrastructure failure may be replayed under a frozen rule before any metric is read.
7. Freeze initialization per architecture. Equal tuning counts are fair, but source-faithful Tropical Attention and exact-max models cannot be left to an implementation-time interpretation of “Xavier-uniform linear initialization.”

Every added control changes the test count, critical value, power calculation, training-run count, and control-stage order; update all of them together.

#### 3. Fix the estimand and simultaneous inference

1. The current estimand, the arithmetic mean of seedwise ratios `1-e_candidate/e_control`, is unstable when a control seed has small error, is unbounded below, and is not the usual relative reduction in expected error. A Student-t interval with 16 seeds does not become valid merely because the critical value is Bonferroni-adjusted. Freeze either (a) a ratio of paired seed-mean errors with a prespecified Fieller/delta or cluster-bootstrap simultaneous interval, or (b) the mean-of-ratios estimand together with a denominator floor and coverage simulation for the exact discrete-error regime. State explicitly whether the macro contrast is the ratio of the two four-slice macro errors or the mean of four slice ratios.
2. If any control error can be zero or nearly zero, define the inference rule without an implementation-time pseudocount. “Both zero means zero improvement” does not solve nonzero but one-mistake denominators.
3. The 49 count is arithmetically correct only for the current nine controls and only if the two soft-bit tests are primary-macro tests. Define their domain aggregation. If the one-point protected margin must hold on each OOD slice, include those slice statements in the family. Recompute the family after adding controls.
4. Define paired non-inferiority estimates and standard errors as explicitly as capability estimates. Use absolute error fractions internally; “one percentage point” is `0.01`.
5. The quoted 86.6% is power for one idealized macro contrast, not for passing nine controls, 36 slice conditions, two non-inferiority families, and health gates. Label it per-contrast power and add a joint/minimum-contrast sensitivity table or simulation. The slice nonnegative gates may dominate power even when the headline macro effect is 35%.

Bonferroni itself is conservative and defensible once each component interval and the final family are frozen; the blocker is the undefined/fragile component statistic, not the use of 49-way familywise control.

#### 4. Make branch use a reproducible causal test

1. Define lesions as inference-only, with no retraining: zero the selected normalized group after group RMS and before `W_O`, in all heads and both blocks. Freeze the evaluation slices, seed aggregation, and ratio/zero rules.
2. Require a **double dissociation**, not only matching damage: soft lesion must damage `y_s` materially more than `y_m`, and max lesion must damage `y_m` materially more than `y_s`, under frozen absolute or relative thresholds. Otherwise both branches may redundantly encode both subproblems and still satisfy the present rule.
3. Define score shuffling exactly: query-score rows only, max branch only, within each fixed-`n` slice, with one stored cross-example permutation per training seed; values, labels, masks, and soft scores remain paired. State whether the effect is a mean across seeds or must hold per seed.
4. For every health condition, specify block/head/channel/query location and aggregation across 16 training seeds and five slices. “At least 75%,” “for every active max channel,” and “mean norm” currently lack these indices.
5. Define gradient health as a gradient with respect to a named tensor or parameter, under a named loss, model mode, reduction, batch partition, and accumulation convention. The `1e-8` threshold is meaningless until sum-versus-mean scaling is frozen.
6. Define winner entropy's normalization (`log n`), float tie handling, and whether every-position support is checked per seed, block, head, channel, and slice. Define isolated `W_O` contribution before/after residual and whether the reported quantity is a mean of ratios or ratio of means.

These health gates need not enlarge familywise type-I error if they are preregistered intersection gates that can only reject. They do need deterministic definitions; they are not deterministic merely because their thresholds are called integrity kills.

#### 5. Make CPU-time matching real

1. Calibrating only at batch 128, sequence length 9 does not match the L16/L32 primary workload where reducer costs diverge. Freeze a weighted workload over the primary lengths (and report each length separately), then select from it.
2. `r_time=1` may leave a large candidate runtime premium unspent because only integer repeated depth is available. Freeze a finite menu that can spend residual time through tied recurrence, FFN width, head layout, or additional ordinary layers, and require the selected control to land inside a two-sided band such as 95--105% of candidate p50. If none lands in-band, call the contrast unmatched; do not label it time-matched.
3. Freeze whether p95 must also be non-inferior, how models are isolated in fresh processes, CPU affinity/frequency policy, and how RSS is measured without allocator contamination. Specify whether “trained timing” means one frozen seed per architecture or all 160 trained models; this ambiguity changes the protocol cost substantially.

#### 6. Repair hash and acceptance sequencing

The current order hashes the draft and then edits the same file to mark it `ACCEPTED`, immediately invalidating the hash. It also chooses the official Tropical Attention revision only after implementation. The safe order is:

1. create `v2` containing all audit-driven semantic changes and immutable upstream commit/environment identifiers;
2. obtain both audits of that exact byte sequence;
3. record acceptance in a separate immutable sidecar, without editing `v2`;
4. hash `v2`, both audit decisions, the upstream/vendored source, environment lock, fixed mixing matrix, and fixed evaluation data;
5. implement; then hash all source and structural manifests;
6. obtain the independent source-level `RUN CPU` decision before any training or accuracy artifact is decoded.

Also freeze the complete training-data seed schedule before execution; fixed evaluation hashes alone do not guarantee paired training streams.

### Is this still a cheap local falsifier?

For the current ten architectures (candidate plus nine controls), the frozen core already entails:

- 20 tuning runs and 30,000 optimizer steps;
- 160 evaluation-training runs and 240,000 optimizer steps;
- about 34.6 million training examples at batch 128;
- eight million base model-example evaluations from 160 models times five 10,000-example slices, plus roughly two million candidate lesion/shuffle evaluations;
- leakage models, Stage 0, and at least millions more timed batch-examples depending on what “trained timing” means.

Added fatal controls increase those counts. The models are tiny, so this remains local and bounded, but under one-thread deterministic CPU it is plausibly hours to low days, not a trivial smoke test.

Add a preregistered staged no-rescue order. Run operator/generator/source checks first; then candidate health; then complete all 16 seeds for the strongest fatal absorbers (group-normalized narrow softmax, independent-score hybrid, all-max T84, and faithful Tropical Attention) in a frozen order. After each **complete** contrast, an independent decoder may reveal only `STOP-REJECT` or `CONTINUE`; any failure stops v1, and no threshold, seed count, control, or hyperparameter changes. Do not take interim looks within 16 seeds unless an alpha-spending/futility rule is added. Keep the final simultaneous family based on all planned tests for any protocol that reaches a positive decision.

### Decision

The mechanism-facing generator is worth preserving, but v1 still permits outcome-relevant choices and contains OOD leakage, unfair crash semantics, incomplete controls, an unstable inference contract, underspecified branch tests, non-matched timing, and a self-invalidating hash order. Do not accept or implement it. Issue and independently re-audit `v2`.

REVISE PREREGISTRATION

## Exact v2 preregistration re-audit

Date: 2026-08-02  
Object: `shared-score-heterogeneous-attention-t84-cpu-preregistration-v2.md`  
Claimed SHA-256: `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`

### Exact checks that pass

- The v2 file hash is exactly `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`.
- The mixer file hash is exactly `f84aa219820c84314da2eede2e0e8b45bc83c881a1b01d2b1d6973d1cd8537c8`. It contains 196 hexadecimal binary64 values; their canonical row-major little-endian bytes hash to `2c1ac147b8f896e889695097fa732e2811ec8aeabe45fddc211482f08a45c973`. Its maximum observed `|Q^TQ-I|` in binary64 is `6.66e-16`.
- The wrapped Tropical Attention snapshot hashes to `af11263442321708748c06cb41c1493bf8da9f15e1e6099aaa293008ed03ef51`. Removing its first six lines yields exactly the stated upstream-file hash `5d06665382adc632d63028f7eab3b1c62e84ca183c6d0fb594813da8a19a7068`.
- The frozen interpreter reports Python 3.14.4, NumPy 2.3.5, PyTorch 2.13.0+cpu, and mpmath 1.3.0; the CPU model is the stated Intel Core Ultra 9 275HX.
- The parameter arithmetic is correct. The standard model is `448 + 2*(64+4096+4096) + 32 + 198 = 17,190`. The official-tropical block is `64+1,216+6,976=8,256`, preserving 17,190 total. The conditioned-DeepSets control sums to 17,090.
- Thirteen architectures imply 26 tuning jobs, 208 evaluation jobs, 351,000 optimizer steps, and 44,928,000 batch examples. The stated 221,184,000 maximum tied-sublayer/example applications is also arithmetically correct.
- The 104-way family count is correct: `60+12+12+3+3+12+2`. The one-sided `0.05/104`, 15-df critical value is correctly frozen at `4.092099746816106`.

V2 resolves most substantive v1 defects: OOD tuning leakage is removed; the final-query mask is exact; fixed evaluation supports and training pairing are defined; all-max, grouped-narrow, independent-score, whole-head, official-tropical, and nonlinear pooled controls are present; normalizations and parameter counts are explicit; a control crash cannot benefit the candidate; the bounded paired `0.8 e_control-e_candidate` estimand replaces the unstable seedwise ratio; lesion and health indices are mostly frozen; length-33 and p95 enter time matching; the complete family and high-false-negative interpretation are honest; staging is no-rescue; and acceptance no longer edits an already-hashed preregistration.

### Remaining semantic choices require v3

#### 1. The leakage-probe training corpus has no seed

Section 5 standardizes and trains on a “100,000-example train” set, but no seed or construction maps that corpus to section 4. The `8402xx` values seed probe models/index permutations, and `840301/840302` schedule MLP indices; none specifies the latent examples. This is outcome-relevant because a different corpus changes standardization, leakage accuracies, and whether model training is authorized.

V3 must freeze a probe-fixture seed, the exact same latent draw/rejection/permutation procedure, whether it is exactly quadrant balanced, and a pre-training fixture hash. The batch cycling should be stated algorithmically as one infinite index stream: consume the current permutation remainder, concatenate exactly the needed prefix of the next independent permutation, advance that next permutation's cursor, and never repeat or drop an index at the boundary. The current “carried into the front” language is close but not byte-exact.

#### 2. Timing inputs and orchestration are undefined

“Fixed contiguous float32 inputs from seed 843000” does not specify RNG family, distribution, tensor shape (`14` raw/mixed token fields versus `32` hidden features), whether the full model or a block is timed, token/mask construction, or initialization seed. These choices can select a different `soft_time_matched` control. “Fresh process per variant” also conflicts with an unspecified “20 round-robin blocks”: it does not say whether one process persists for 20 blocks, processes are relaunched per round, or variants are interleaved.

V3 must freeze the exact timing tensor bytes or their generation and hash, the callable boundary, initialization, mask, per-forward timing sample, variant/process order for all 20 rounds, cache/warmup handling, and whether `VmHWM` means raw high-water mark or baseline-subtracted value. The current 95--105% eligibility and lexicographic selection rule can then remain unchanged.

#### 3. The eight-hour resource gate lacks a ledger

The arithmetic ceiling is clear, but “eight single-core hours” does not define clock source or which work is charged. V3 must state whether it is the sum of monotonic wall time or process CPU time and explicitly include/exclude fixture generation, timing repeats, Stage 0, probes, tuning, training, fixed-slice evaluation, perturbations, health backward passes, hashing, decoder time, and an allowed infrastructure replay. Without this, the same artifacts can be called either complete or `RESOURCE-INCONCLUSIVE` after their cost is known.

#### 4. Initialization and optimizer details still depend on future source

“Modules are created in the frozen table order in source” freezes nothing before the source exists. Random draw order can change optimization, especially for the exact-max and upstream-random controls. V3 should derive an independent seed for every named parameter tensor from `(training_seed, architecture_id, block, module, tensor)` or enumerate one canonical creation order in the preregistration. It should also state that AdamW's single parameter group contains all trainable parameters (or enumerate exclusions) and freeze the exact global-gradient clipping operation, norm type, and placement between backward and optimizer step.

#### 5. Block-two route shuffling is not executable from the prose

The stored Sattolo map is global over 10,000 examples, not generally within an evaluation batch. After block-one score shuffling, block-two scores depend on already-perturbed hidden states. V2 does not say whether block-two donor rows are computed from intact states, from each example's layerwise-perturbed state, or from a jointly cached full-slice pass. These are different interventions and can change lesion effects.

V3 must freeze a layerwise algorithm and batching/cache contract. The natural choice is: cache all current block-one score rows, permute the named branch, complete block one for all examples, cache block-two score rows from those perturbed states, apply the same stored example map, then complete block two; no within-batch replacement is allowed. If intact donor rows are intended instead, say so explicitly.

#### 6. Infrastructure replay and decoding need closed definitions

The remaining result-dependent architecture/LR/timing choices are otherwise appropriately frozen: LR uses only in-support development data, time-control selection sees timing only, and every accuracy look occurs after a complete 16-seed stage with the final 104-way critical value. However, “common infrastructure failure” still needs a whitelist and receipt rule so a semantic/numerical control failure cannot be reclassified after logs are seen. The decoder source and its mapping from each completed receipt to the four allowed words should be hashed before Stage 0, not first defined during staging.

### Mechanism isolation and fairness

The frozen task and control triangle are adequate for the narrow **architecture-allocation** claim once the execution ambiguities above are removed. Every instance needs both independent bits over one latent score; `hybrid_whole_head` holds rank/head size while removing within-head heterogeneity; `hybrid_independent` gives separate score maps at fixed total QKV width; and `all_max_shared`, grouped narrow heads, official tropical attention, and conditioned DeepSets close the main alternate explanations. Global branch lesions, score shuffles, absolute effects, and double dissociation prevent a reducer-level bypass.

This does not prove that each individual head uses its one score productively for both groups. If the intended conclusion is stronger than the frozen architecture-level claim, add a per-head paired-use intervention; otherwise explicitly retain the present architecture-level wording in every result.

### Cost and staging

The protocol is no longer especially cheap: its unfalsified ceiling is 234 training jobs, 351,000 optimizer steps, 44.9 million training examples, 10.4 million base evaluations, plus perturbations, health, probes, and timing. The frozen complete-contrast order is a sound reject-only strategy: no within-seed peeking occurs, every decoded contrast uses the final family critical value, and a failure can only stop rather than rescue the lane. Once the eight-hour ledger is defined, no additional statistical staging edit is required.

### Decision

The scientific design is substantially repaired, and all stated hashes and arithmetic verify. But the missing probe corpus seed, timing workload definition, resource ledger, initialization order, and block-two shuffle semantics are outcome-relevant implementation choices. Under the instruction that any semantic ambiguity requires a new version, the exact frozen v2 cannot be accepted or edited in place.

REVISE V2

## Effective v3 ordered-pair audit

Date: 2026-08-02  
Base: v2 SHA-256 `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`  
Overlay: v3 SHA-256 `b10c376e386c916bfb25bc11a0af2ef3f2433439b99d98c5e5515af2047b6bbf`

### Verified repairs

Both files match the stated hashes and their precedence rule is explicit: v2 remains normative except for text replaced by v3. The overlay closes the named v2 blockers in substance:

- The 100,000-example probe corpus now has its own seed, exact quadrant collection, authoritative fixture, and an explicit no-drop/no-duplicate permutation stream.
- Timing now uses hashed valid mixed-token batches at all three lengths, a full-model callable, exact warmup/sample counts, a rotated 20-round fresh-process schedule, absolute `VmHWM`, and a charged retry.
- The resource section identifies monotonic wall time and enumerates nearly all charged work.
- Tensor-local initialization, one ordered AdamW group, weight-decay scope, optimizer flags, clipping, and optimizer-step order are specified.
- Route shuffling is full-slice, recursively propagated into block two, and forbids minibatch donor substitution or intact block-two donors.
- The two competence gates bring the family to `60+12+12+3+3+12+2+2=106`. The frozen one-sided tail probability is exactly `0.05/106`, and `tcrit=4.1014945453569363208` evaluates to that tail for 15 degrees of freedom.
- Replay discretion is removed, the decoder vocabulary is closed, and acceptance binds the ordered hash pair without editing it.

The parameter counts and job/resource arithmetic verified in the v2 audit are unchanged and remain correct. The scientific task/control design still isolates the narrow architecture-allocation claim rather than a broad attention or reasoning claim.

### Remaining effective-protocol ambiguities

Under the instruction that any semantic ambiguity requires another version, the ordered pair is not yet executable.

#### 1. Variable-length training batches have no tensor/execution contract

V2 draws `n` independently in `4..8` inside each exactly balanced batch of 128, while also saying there are no padding tokens. Such examples cannot form one ordinary dense token tensor. Neither v2 nor v3 says whether to pad and mask, loop per example, or group the nominal batch into length-homogeneous microbatches. Those choices change numerical reduction order, wall time, and gradients.

The next overlay must freeze one method. A clean no-padding contract is to group the 128 examples by `n`, run the five length-homogeneous microbatches in ascending `n`, restore original example order, form one 128-example mean loss, call backward once, and take one optimizer step. If backward is performed per microbatch instead, the exact `microbatch_size/128` loss weighting and accumulation order must be stated. The in-support development set and variable-length probe fixture need the same ragged-storage/evaluation rule.

#### 2. V3.2 accidentally replaces the time-control architecture definition

V3.2 says it replaces the first three paragraphs of v2 section 8. Paragraph two is the only definition of what `r_attention` and `r_ffn` do: tied repeated pre-norm attention and FFN residuals in each block. V3 selectively retains the eligibility/weighting/MAD rules from paragraph three and assigns IDs to `(r_attention,r_ffn)`, but never explicitly retains or restates the repeated-block semantics it replaced.

The overlay must say that v2's tied-repetition architecture paragraph remains normative, or restate it verbatim. Otherwise the 16 timed/trained variants have IDs but no operator.

#### 3. Tensor paths still permit implementation-chosen seeds

V3.4 makes initialization independent of construction order, but the seed hashes the literal `canonical_path`, and no authoritative path table or grammar is included in the ordered pair. An implementer can rename a module or tensor and obtain a different initialization while satisfying “contains architecture ID, block, module, tensor.” The later source audit observes that choice after it has been made; it does not preregister it.

Freeze the complete path table for every architecture/probe, or define a closed grammar with enumerated architecture/module/tensor IDs and indices. UTF-8 bytes, decimal seed formatting, separators, and absence of aliases should be explicit. Generated static manifests can then verify, rather than choose, the seeds.

#### 4. The Sattolo permutation itself is not reproducible

V3.5 freezes how a stored `pi` is consumed but inherits only “a stored Sattolo-cycle derangement from PCG64 seed” from v2. Sattolo implementations differ in loop direction, integer endpoint convention, and permutation orientation. No permutation exists in the frozen pair yet, so its later fixture hash does not remove implementation discretion.

Specify the exact algorithm, for example starting with `[0,...,9999]`, iterating `i=9999..1`, drawing `j=PCG64.integers(0,i)` with the upper endpoint excluded, swapping positions `i,j`, and interpreting the resulting array as receiver-to-donor `pi(b)`. Then hash the four-slice/16-seed/two-branch permutation fixtures before perturbation execution.

#### 5. Decoder timing conflicts with its immutability rule

V3.7 requires the decoder to be independently audited and hashed “before Stage 0,” while its final sentence says its source/hash cannot change after the **first execution authorization**. Under the inherited v2 sequence, `FIXTURES AND TIMING` is authorized and executed before Stage 0. V3.8 adds the decoder only to the pre-Stage-0 manifest. Thus the decoder is declared immutable from a point at which it need not yet have an audited hash, and the timing receipt/time-control selection occurs before the closed decoder is guaranteed.

Move decoder source audit/hash into the manifest required before `FIXTURES AND TIMING`, and state that the same decoder handles timing/MAD/resource receipts from the first executable stage onward.

#### 6. Resource accounting needs a non-overlap rule

V3.3 charges “controller invocations and their workers” and records each invocation from its first operation through its last hash. If worker durations are also ledger entries, their time is double-counted inside the enclosing controller span; if only controller spans are entries, it is counted once. Freeze the latter or an explicitly disjoint interval union. Also define artifact use as the sum of final logical byte lengths under one artifact root, whether temporary/deleted failed-stage files count, and when the 2 GiB measurement is sampled. These choices can change `RESOURCE-INCONCLUSIVE` after cost is observed.

### Result-dependent choices and fairness

The accuracy-facing rules are otherwise closed. Development LR selection is in-support only; timing alone selects the time control; competence, causal, floor, absolute, and relative gates all use the final 106-way critical value; and complete 16-seed stages can only continue or terminate, never rescue a model. Control failure cannot improve the candidate. No new statistical or control-family revision is required beyond the semantic repairs above.

### Decision

The overlay fixes every *named* v2 concern conceptually, and its hashes, family count, critical value, controls, and arithmetic verify. But the effective pair still leaves variable-length batch execution, the time-control operator, initialization seeds, derangement generation, decoder sequencing, and resource accounting to future implementation choices. Those are outcome-relevant and cannot be repaired by source audit under the stated strict standard.

REVISE V3

## Final effective v4 ordered-triple audit

Date: 2026-08-02  
Base: v2 SHA-256 `a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5`  
First overlay: v3 SHA-256 `b10c376e386c916bfb25bc11a0af2ef3f2433439b99d98c5e5515af2047b6bbf`  
Final overlay: v4 SHA-256 `085abc5e703ee4a064a837f66cf3ffb82e282c1fe78fec12cc53870a0db774b8`

### Verification

All three files match the stated hashes. Their precedence is unambiguous: v2 is normative, v3 replaces conflicting v2 text, and v4 replaces conflicting v2/v3 text. The underlying versions remain immutable.

V4 closes each of the six blockers from the effective-v3 audit:

- Training is now composed of homogeneous, no-padding batches with one frozen length per step. Development and probe fixtures are fixed as five explicit length buckets with exact counts and aggregate denominators.
- The time-control variants again have a complete operator: each block repeats the pre-norm attention residual and then the feed-forward residual, with weights tied across repetitions within that block and not across blocks.
- Every trainable tensor has a closed canonical parameter path under an exhaustive architecture/module/tensor grammar. The later source audit can only verify the sorted path/shape manifest; it cannot choose initialization seeds by naming implementation modules differently.
- The stored derangement is generated by an exact PCG64/Sattolo loop with frozen bounds, swaps, and receiver-to-donor orientation, plus permutation and fixed-point assertions. Its recursive two-block use remains the v3 intervention contract.
- Decoder source, tests, and hash must now pass independent audit and enter the initial manifest before `FIXTURES AND TIMING`; the decoder cannot first be defined after timing receipts exist.
- Resource time is the top-level controller interval counted once. Worker durations are diagnostic only and cannot be added to the charged total. The artifact root, recursive allocated-byte formula, logical-byte report, memory ceiling, append-only behavior, and invalid killed/incomplete-controller case are all frozen.

The inherited scientific and statistical contract is unchanged. The family still contains 106 tests, uses one-sided `alpha=0.05/106`, and freezes `tcrit=4.1014945453569363208` at 15 degrees of freedom. The control triangle, competence gates, perturbations, complete-stage stopping rules, and narrow architecture-allocation interpretation therefore carry forward without another multiplicity or fairness revision.

### Conflict and discretion check

V4's replacements are scoped cleanly. The homogeneous-batch rule deliberately supersedes the earlier per-example length draw and is compatible with the no-padding mask contract. The repeated-residual clause restores the operator accidentally removed by v3 rather than defining a second time-control family. The canonical path grammar, Sattolo orientation, pre-timing decoder closure, and single-count resource ledger each remove implementation discretion without changing the registered scientific contrast.

No conflicting inherited clause remains effective, and no result-dependent architecture, seed, timing, decoder, resource, or intervention choice is introduced by v4.

### Scope of acceptance

This is acceptance of the effective preregistration only. It is not an acceptance of future source or receipts, and no implementation or experiment was run for this audit. The registered source audit, fixture/timing authorization, staged execution gates, hashes, and resource ceilings remain mandatory exactly as specified by the effective ordered triple.

### Decision

ACCEPT EFFECTIVE V4
