# Outward mathematics reset — one epsilon separation, zero model candidates

Status: **complete; no candidate 004 admitted**  
Date: 2026-07-26

## Direct answer

The pass did not prove a new generally smarter model at unchanged served
resources. It proved why that target cannot follow from algebra alone, and it
found one honest different-currency result: for constant or inverse-polynomial
`epsilon`, private randomized fingerprints exchange exactness for exponentially
less state on equality.

This is not a failure to invent enough operators. Every finite-precision
operator we tried is itself a program. Once the control receives the same
memory, instructions, randomness, and error budget, it can execute that program
exactly.

## Matched-machine inclusion theorem

Let architecture `A` have:

- `P_s` static bits;
- `S` mutable request bits;
- at most `T` charged finite-precision primitive operations per token;
- a fixed compiled inference program.

Construct recurrent RAM controller `R_A` with the same static bits, mutable
bits, compiled operation sequence, instruction/addressing set, and—for a
randomized `A`—the same coupled random tape. At token zero their states agree.
If the states agree before instruction `j`, executing the identical instruction
on identical operands makes them agree after instruction `j`. Induction over
the `T` primitive invocations and then over tokens gives identical outputs and
states.

If architecture code is charged, charge the same program to both. If fixed
system code is excluded, exclude it symmetrically. Therefore

\[
\mathcal F_A(P_s,S,T)\subseteq \mathcal F_{recurrent\ RAM}(P_s,S,T).
\]

Here `T` is matched logical work, not a claim of equal wall-clock latency across
different hardware paths. No new algebra can prove strict dominance over that
matched control. A claimed edge must instead name something the control was
denied: an addressing mode, instruction, random bits, nonzero error, workload
prior, larger code, external state, different precision, training advantage,
or a physically cheaper hardware path.

This theorem does **not** say Transformer, Mamba, and MoE are equally good in
practice. Their training geometry and hardware execution differ. It says those
differences must be measured; they cannot be inferred from a new mathematical
notation for an already executable program.

## Worlds tested

| Outside world | Valid theorem | Fatal control | Decision |
|---|---|---|---|
| Quotients and group actions | Weighted union-find stores exact monotone group relations in state within a constant factor of the information bound | Same random-access recurrent controller; direct ED-Batch Algorithm 5 collision | Classical typed data structure |
| Exterior/Grassmann algebra | A pivoted XOR basis stores an exact subspace near its `r(D-r)`-bit lower bound | The factorized blade is incremental Gaussian elimination; a full multivector costs `2^D` coefficients | Classical linear-closure primitive |
| Free groups/topological cancellation | For free-group rank `k >= 2`, a stack stores the unique reduced word within a constant factor of the exact streaming-state lower bound; rank one reduces to a logarithmic-bit signed counter | It is exactly a pushdown automaton with the same stack | Classical stack primitive |
| Symmetry/orbit memory | Canonicalization stores one payload per known orbit rather than one per transformed copy | Equivariant weight tying and canonicalize/undo are identical; no new independent information | Distributional compression only |
| Sparse syndromes/sketches | Sparse structure can use `O(s log(U/s))` measurements | Succinct sparse dictionaries meet the same support-entropy bound without recovery error | Approximation/engineering trade |
| Adaptive foveated precision | Variable bit allocation beats a naïve axis-uniform quantizer on sparse-axis states | A `B`-bit state still has only `2^B` configurations; selectors, scales, and directories consume the apparent slack | Average-distortion trade only |
| Packed Boolean presence plane | `M` bytes hold `8M` exact predicates, 16x the channel count of BF16, with word-parallel OR updates | Both formats contain the same `8M` physical bits; an arbitrary independently specified dense mask costs `Omega(8M)` Boolean decisions or an `M`-byte table, while sparse/structured masks restrict the feature family and need addressing | Typed systems sidecar only |
| Probabilistic certificates | A verifier may cheaply detect some bad proposals | Without retry, interaction, or extra candidates, it adds no task information beyond `(A,C)`; any decoded correction belongs to the composed circuit available to the matched control | Reliability/search tool, not free capability |
| Private finite-field fingerprints | Equality uses `O(log n + log(1/epsilon))` state instead of the deterministic exact `Omega(n)` bound | A randomized digital recurrence is identical; Karp–Rabin is classical | **Valid epsilon/state trade**, not a model candidate |

## Strict logical-state separation: deterministic exact versus randomized error

For equal-length strings `x != y`, evaluate their token polynomials at a fresh
private random `r` in `F_p`. The difference is a nonzero degree-below-`n`
polynomial, so

\[
\Pr[h_r(x)=h_r(y)]\le(n-1)/p.
\]

At one million byte tokens, the frozen 64-bit construction uses 234 logical
bits versus an 8,000,000-bit deterministic exact lower bound, with one-sided
error below `5.5e-14`. This is a word-RAM logical-state separation, not a served
Pareto result: RNG, packing, encoding, and physical modular arithmetic remain
unmeasured. The executable gate exhaustively checked 1,370,838 small-field
evaluations and exercises an actual bounded streaming phase machine through the
million-token boundary. See
[`epsilon-fingerprint-register-stage0.md`](epsilon-fingerprint-register-stage0.md).

The gain is real but narrow: it retains equality, not token content or semantic
meaning. Calling it a smarter language model would be false.

## No-new-information and matched-composition observation

Let a generator emit `(A,C)` and a verifier see only `(X,A,C)` plus independent
randomness `R`, with no extra search or interaction. Returning `A` leaves
accuracy unchanged; abstaining trades coverage for precision; generating a
replacement answer creates a larger composed answer-producing circuit that the
matched control also receives. If `R` is conditionally independent of `Y` given
`(X,A,C)` and `hat Y=f(X,A,C,R)`, data processing gives

\[
I(Y;\hat Y\mid X)\le I(Y;A,C\mid X).
\]

Postprocessing can improve accuracy by decoding information already present in
`(A,C)`; it does not add task information. The same postprocessor, operations,
and state must therefore be given to the matched direct model.

Interactive proofs and verifier-guided LLM methods gain from resources excluded
here: stronger provers, messages, candidates, retries, or extra passes. Freivalds
can cheaply reject a wrong matrix product; it does not construct the right one.

Primary boundaries:

- [Karp–Rabin fingerprints](https://doi.org/10.1147/rd.312.0249)
- [Freivalds verification](https://doi.org/10.1007/3-540-09526-8_5)
- [ED-Batch weighted transformation union-find](https://proceedings.mlr.press/v202/chen23g/chen23g.pdf)
- [Neural Stacks](https://proceedings.neurips.cc/paper/2015/file/b9d487a30398d42ecff55c228ed5652b-Paper.pdf)
- [Modern equality randomness/error tradeoffs](https://eccc.weizmann.ac.il/report/2025/068/)

## Research consequence

No GPU should be rented for these branches. The next architecture idea is
admissible only if it declares the different currency before its mechanism:

1. accepted epsilon/randomness;
2. a measured GPU instruction or bandwidth asymmetry;
3. a distributional structural prior with a randomized-mapping control; or
4. a learnability advantage under equal data and training compute.

Without one of those, another graph, manifold, energy, group, stack, sketch, or
code is only a different program for the matched controller.
