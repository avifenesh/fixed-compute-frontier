# Capability Contract v0

## Claim scope

Every submission declares one accounting scope before training:

- **Serve-only:** training cost may move but is fully disclosed.
- **Lifecycle:** training plus amortized serving over a fixed request volume is
  capped.

Quality is a vector, never one average. A candidate must be noninferior on every
protected slice and strictly improve at least one preregistered coordinate.

## A-E decisions

### A — Smaller model with larger-model capability

Maintain a tuned baseline ladder of at least three adjacent sizes. The candidate
must match every protected capability slice of a larger point while using fewer
exact checkpoint bytes. Report any FLOP, state, or latency increase as a trade.
No extrapolation beyond the measured size ladder.

### B — Faster model with slower-model capability

First establish quality equivalence/noninferiority. Then compare TTFT, TPOT,
completed requests per second at the SLO, and total time per naturally
terminated request. Forced-length decoding is diagnostic, not sufficient.

### C — Light model with dense knowledge

Never combine these three measurements:

1. **Incompressible storage:** private random key-to-uniform-value mappings;
   report cross-entropy-derived accessible bits, exact recall, and the capacity
   curve as fact count grows.
2. **Structured reuse:** accuracy on unseen combinations from a private latent
   world; this measures composition, not stored independent bits.
3. **Natural-like knowledge:** recall and calibration over a private injected
   micro-corpus with aliases, paraphrases, head/tail frequencies, conflicts, and
   exceptions.

Report raw knowledge, knowledge per exact checkpoint byte, peak served HBM, and
decode cost. Protected general capability must remain noninferior.

### D — Less GPU cost with more performance

Use the actual deployment surface. Compare gross GPU-seconds/dollars and joules
per completed request under identical latency SLO, workload, batching, context,
decoding, and quality. Include CPU, network, helpers, and all allocated GPUs.

### E — Strict dominance

E passes only if every protected capability and served-resource coordinate is
noninferior, at least one clears a strict practical margin, and the result holds
across at least three training seeds and two model scales, including one
deployment-relevant scale.

## Protected capability slices

- **P1 General modeling:** bits per raw byte and NLL over source-clustered prose,
  code, math, and multilingual data.
- **P2 Reasoning/composition:** executable code, exact math/algorithms, unseen
  factor combinations, and difficulty/length extrapolation.
- **P3 Knowledge:** head/tail closed-book recall, calibration, random-bank
  accessible information, and post-interference retention.
- **P4 Robustness/behavior:** paraphrase consistency, counterfactual
  sensitivity, calibration, repetition, instruction following, and refusal as
  appropriate to base or post-trained models.
- **P5 Context/output:** short/long-context use, correctness under a fixed output
  cap, and natural output-length distribution.

Base and post-trained claims are separate. A deployable chat model adds a
mandatory safety pack.

## Evaluation packs

- **PUBLIC_DEV:** iteration only; never proves A-E.
- **PRIVATE_CORE:** locked, source-clustered natural text/code/math/multilingual
  data, primarily scored by bits per raw byte plus task metrics.
- **PROCEDURAL_PRIVATE:** secret-seed algorithms, composition, context use, and
  extrapolation generated after artifact freeze.
- **KNOWLEDGE_LAB:** matched random, structured, and natural-like injection
  banks with acquisition and retention curves.
- **SERVICE_TRACE:** frozen prompt, context, output-length, arrival, and SLO
  distribution.

## Random and contamination controls

- Generate random mappings from a committed cryptographic PRF; match structured
  and random banks for token lengths, frequencies, exposures, and interference.
- Freeze artifact hashes before revealing evaluation templates and seeds.
- Split procedural tasks by generator family and latent graph, not random rows.
- Scan training and teacher corpora for exact and semantic near-duplicates.
- Block teachers, retrieval, and tools from locked evaluation unless their full
  cost belongs to the declared system.
- Public development and locked adjudication sets never overlap; locked-score
  feedback is rate-limited.

## Statistics

- Two paired seeds may reject at an early gate; they cannot prove a claim.
- Main claims require at least three independent training seeds; use five when
  the effect is under twice baseline seed variation or near a margin.
- Bootstrap at the dependency unit: source document, latent world, repository,
  problem family, or service-trace window—not individual correlated tokens.
- Use simultaneous one-sided 95% intervals or Holm correction. Report every
  seed, mean, median, and worst seed; never select the best seed.

## Scale gates

1. **G0 Accounting/correctness — minutes:** artifact hash, exact bytes, graph,
   analytic resource ledger, numerical sanity.
2. **G1 Cost feasibility — hours:** target-shape whole-block and short
   end-to-end run on the real target GPU.
3. **G2 Information sanity — hours/day:** tiny model, matched structured/random
   loads, protected mini-suite, two seeds. This can reject only.
4. **G3 400M class — days:** three seeds, full private suite, service trace, and
   adjacent tuned baseline sizes. First publishable mechanism signal.
5. **G4 Scale replication — >=1B:** repeat on a deployment-relevant shape. No
   strict A-E claim before this gate.

Never extrapolate across a kernel, parallelism, or memory-capacity regime
change. A crossover outside tested scales is a hypothesis, not a result.
