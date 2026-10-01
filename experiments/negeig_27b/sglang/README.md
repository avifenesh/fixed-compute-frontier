# Widened-beta gate (negeig) on SGLang 0.5.20

Serves Qwen3.8-27B (`qwen3_5`, 48 Gated DeltaNet layers + 16 attention layers) with the trained
beta gate, for evaluation and for RL rollouts. No SGLang file is edited on disk. The gate is a runtime
plugin.

Per GDN layer, per value head:

    s    = sigmoid(b)
    t    = tanh(W x)               W: fp32 [num_v_heads, hidden], zero at start
    beta = s + t * (2 - s if t >= 0 else s)

At W = 0 the model is the stock model bit for bit. beta reaches 2 as t reaches 1, and 0 as t reaches -1.
The HF training side is `experiments/negeig_retrofit/patch.py` (arm `wide`) and `experiments/negeig_27b/train27.py`.
This directory does not touch either.

Pinned: SGLang 0.5.20, checkpoint `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
transformers 5.17.0 for the HF reference.

## Files

| file | what |
|---|---|
| `negeig_sglang.py` | the plugin, plus a CLI: `static-check`, `export-gates`, `install-plugin`, `merge-lora` |
| `check_parity.py` | the GPU-box parity test: `prepare`, `make-random-gates`, `collect`, `hf-ref`, `compare`, `selftest` |
| `test_negeig_cpu.py` | 106 CPU checks with a stubbed SGLang model (no GPU, no SGLang install) |

## How the patch works

* `negeig_sglang:register` is a `sglang.srt.plugins` entry point. `SGLANG_PLUGINS=negeig` loads it in every
  scheduler process before the model is built.
* `Qwen3_5GatedDeltaNet.__init__` is wrapped: each layer gets a fp32 parameter `negeig_w`
  `[num_v_heads / tp, hidden]`.
* `Qwen3_5GatedDeltaNet.forward` is rebuilt from its own source with two inserted lines. One calls the producer on
  `b` before the backend call. The other turns off the fused decode projection for gated layers. The patch
  refuses to install if either anchor line is missing or occurs twice.
* The producer is a torch custom op (`negeig::beta`): fp32 `F.linear` at "highest" precision, `tanh`, then a Triton
  combine kernel. Its output is the final beta in the dtype of `b`.
* Two kernels turn `b` into beta with `sigmoid`: the fla `fused_gdn_gating` (prefill, mixed extend) and the packed
  single-token decode kernel. For gated layers both are swapped for source-rewritten copies with the sigmoid
  removed. The delta rule multiplies the value and the state readout by beta, so the final number has to arrive there.
* Paths that cannot take a final beta raise for a gated layer instead of computing a stock sigmoid: the generic
  fused sigmoid-gating decode, target verify, replay-SSM, and FlashInfer or CuTe decode. FlashInfer and CuTe prefill
  are refused unless `NEGEIG_ALLOW_FLASHINFER_PREFILL=1`.
* The fla kernels that read beta do not clamp it or assume beta <= 1. Checked in the 0.5.20 source: beta is loaded
  and used as a plain multiplier (`wy_fast`: `v * beta` and `k * beta * exp(g)`; `chunk_intra`: `A_kk * beta`;
  `fused_recurrent`: `v *= beta`). No clamp, no min or max, no sqrt, no log, no assert on beta.
* On start each rank prints one line to stderr, which `check_parity.py` reads as engagement evidence:
  `negeig: active v1: N gated GDN layers on this rank, gates from <how>, max|W|=<x>`.

## Launch recipe

Install the plugin into a directory (no pip):

    python negeig_sglang.py static-check                       # run inside the SGLang 0.5.20 env
    python negeig_sglang.py install-plugin --dir /opt/negeig-plugin

Launch:

    export PYTHONPATH=/opt/negeig-plugin:$PYTHONPATH
    export SGLANG_PLUGINS=negeig
    export NEGEIG_GATES=/path/to/gates.safetensors      # or: zeros | stream
    python -m sglang.launch_server \
        --model-path /path/to/Qwen3.8-27B \
        --linear-attn-backend triton \
        --linear-attn-prefill-backend triton \
        --linear-attn-decode-backend triton \
        --disable-radix-cache \
        --tp-size 1 --port 30000 2>&1 | tee server.log

Rules:

* The three `--linear-attn-*` flags must be `triton`. The patch raises on a gated layer otherwise.
* No `--enable-linear-replayssm` and no speculative decoding. Both raise on a gated layer.
* Do not pass `--enable-tf32-matmul`. The producer forces "highest" precision around its own matmul, but the flag
  is one more thing to rule out if a number moves.
* `--disable-radix-cache` is for the parity runs, which read full input logprobs. Serving for evals does not need it.
* `--load-format dummy` is unsupported (the gate load needs the real load path).
* To separate graph problems from math problems, run the parity gates once in eager mode first
  (`--cuda-graph-config '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}'`), then with the
  defaults.

`NEGEIG_GATES` modes:

| value | meaning |
|---|---|
| `/path/gates.safetensors` | load after the stock weights, once per model instance |
| `zeros` | W = 0: the stock model. Used by G1 and as the RL start |
| `stream` | every gate must arrive in the weight stream, or the load fails |

Stream keys `...layers.N.linear_attn.negeig_w[.weight]` are accepted in every mode, so RL weight sync reaches the
gate. W has to stay fp32 end to end (the patch refuses a low-precision W unless `NEGEIG_ALLOW_LOWP_W=1`). The
verl weight-sync path must carry it as fp32.

Other environment variables: `NEGEIG_ALLOW_SGLANG_VERSION=1` (install on another SGLang version),
`NEGEIG_ALLOW_NON_CUDA=1`, `NEGEIG_CACHE_DIR` (where rewritten kernel sources are cached).

## Gates from a training checkpoint

    python negeig_sglang.py export-gates \
        --trainable trainable_000600.pt --out gates.safetensors --config /path/to/Qwen3.8-27B/config.json

The output is fp32 `[48, 5120]` per GDN layer under `model.language_model.layers.N.linear_attn.negeig_w`. With
`--config` it checks that the gate layers equal the config's `linear_attention` layers.

## LoRA: merge into the base

The 27B retrofit trains LoRA plus the gate. Decision: merge the LoRA into the base bf16 weights and serve the merged
checkpoint with the gate file. Serving the adapter live does not work here: SGLang 0.5.20's
`supported_lora_modules` does not list `in_proj_ba`, and the fused projection that feeds the gate is turned off
under LoRA.

    python negeig_sglang.py merge-lora \
        --base /path/to/Qwen3.8-27B --trainable trainable_000600.pt --out /path/to/merged \
        --scale 2.0 --rounding stochastic

A plain bf16 merge can drop a small adapter delta (round to nearest lost 86 percent of a tiny delta in the CPU
test). The tool rounds stochastically by default, measures `delta_kept` (the part of the intended delta that the
bf16 weights actually moved), writes `MERGE-RECEIPT.json`, and deletes the output and exits non-zero if
`delta_kept` is below `--min-delta-kept` (0.95). Do not serve a merge that failed this gate.

## Parity runbook (GPU box)

What must hold, and what each gate proves:

| gate | claim | how |
|---|---|---|
| G0 | the stock server repeats itself bit for bit | two collects on stock |
| G1 | the patched server at W = 0 equals stock bit for bit (every prefill logprob, top-k, decode token, decode logprob) | zeros server vs stock |
| G2 | with random W the server matches the HF reference patch within the bf16 gap between two stock implementations | rand server vs `hf-ref` |
| G3 | decode (packed kernel, one token) agrees with prefill at the same positions as well as stock does | decode vs replay logprobs |

Steps (one GPU box, 27B bf16, prompts are token ids so the tokenizer never enters the comparison):

    D=/path/to/Qwen3.8-27B
    O=parity-out; mkdir -p $O

    # 0. inputs
    python check_parity.py prepare --out $O/prompts.json --tokenizer $D          # 32 edge-length prompts + text prompts
    python check_parity.py make-random-gates --config $D/config.json --out $O/rand_gates.safetensors

    # 1. stock server (no plugin: unset SGLANG_PLUGINS and NEGEIG_GATES), log to $O/stock.log
    python check_parity.py collect --url http://127.0.0.1:30000 --prompts $O/prompts.json \
        --out $O/stock.json --tag stock --server-log $O/stock.log
    python check_parity.py collect --url http://127.0.0.1:30000 --prompts $O/prompts.json \
        --out $O/stock2.json --tag stock-repeat --server-log $O/stock.log
    # stop it

    # 2. zeros server: SGLANG_PLUGINS=negeig NEGEIG_GATES=zeros, log to $O/zero.log
    python check_parity.py collect --url ... --prompts $O/prompts.json --out $O/zero.json --tag zero --server-log $O/zero.log

    # 3. random server: NEGEIG_GATES=$O/rand_gates.safetensors, log to $O/rand.log
    python check_parity.py collect --url ... --prompts $O/prompts.json --out $O/rand.json --tag rand --server-log $O/rand.log

    # 4. HF references (no SGLang server on that GPU: bf16 27B needs about 54 GB)
    python check_parity.py hf-ref --model $D --prompts $O/prompts.json --out $O/hf_stock.json --tag hf-stock --mode stock
    python check_parity.py hf-ref --model $D --prompts $O/prompts.json --out $O/hf_rand.json --tag hf-rand \
        --mode $O/rand_gates.safetensors

    # 5. verdict
    python check_parity.py compare --stock $O/stock.json --stock-repeat $O/stock2.json --zero $O/zero.json \
        --rand $O/rand.json --hf-stock $O/hf_stock.json --hf-rand $O/hf_rand.json --out $O/verdict.json

Reading the verdict:

* Exit 0: every gate PASS. Exit 1: a gate FAILED. Exit 2: something is INCONCLUSIVE or SKIPPED, which is not a pass.
* G1 is INCONCLUSIVE when G0 fails and the numbers differ (stock does not repeat, so bit identity cannot be read),
  and when the numbers are equal but the engagement evidence is missing: no `negeig: active` line in the zeros log
  (an unpatched server would also match stock), or a `negeig:` line in the stock log (the stock run was not stock).
  Numbers that differ with a repeatable stock server are a FAIL.
* G2 and G3 are INCONCLUSIVE when the random W moved the HF logprobs by less than
  `max(--effect-min, --noise-factor * stock gap mean)`. A vacuous W cannot pass. Raise `--target-t` in
  `make-random-gates` (the default 0.5 is the std of the pre-tanh gate for a unit-rms input).
* `--skip G2,G3` waives gates and gives verdict WAIVED, which does not block. A gate with missing files is SKIPPED
  and does block.
* The G2 and G3 thresholds (`--gap-ratio 2.0 --gap-floor 0.02 --effect-rel-tol 0.5 --corr-min 0.9
  --top1-slack 0.03 --decode-ratio 2.0`) are starting points. Calibrate them against the stock gap on the box
  before treating a FAIL as a bug or a PASS as proof. The stock gap is printed in the verdict.

## What was verified on CPU

All under a CPU cap (`nice -n 10 taskset -c 8-15`, 2 threads), no GPU, SGLang never installed (the 0.5.20 wheel was
unpacked to read and to import against stubs).

* `negeig_sglang.py static-check` against the unpacked 0.5.20 source: each rewrite target occurs exactly once and
  the rewritten forward compiles (two lines added).
* `test_negeig_cpu.py`: 106 of 106. Covers the arithmetic (bit identity to `s` at W = 0, the two branches, the
  limits 2 and 0), the source rewrites, the forward rewrite, refusal guards (wrong decode backend, target verify,
  replay-SSM, generic fused decode, other SGLang version), gate loading in all three modes, tensor-parallel row
  sharding, key mapping, fp32 guards, `export-gates`, `install-plugin` (plugin discovered through a hand-made
  dist-info), `merge-lora` against peft `merge_and_unload` on a tiny model including the red arm where
  round-to-nearest drops a tiny delta, and the producer against the HF retrofit's `_negeig_beta` (bf16 exact), and the producer plus the rewritten forward
  under `torch.compile(fullgraph=True)` with the eager backend (the custom op is resolved once at install, because
  taking a lock inside a traced forward breaks dynamo).
* `check_parity.py selftest --tinyqwen <dir>`: 28 of 28. A mock SGLang HTTP server exercises `collect` and `compare`.
  Green path G0 to G3 PASS. Red arms that fail as intended: a one-float difference at W = 0 (G1), a flaky stock (G1
  INCONCLUSIVE), W ignored, effect halved, effect sign flipped, noise far above the stock gap (G2), a decode path
  that ignores W (G3), a server returning too few logprobs, edited prompt files, record files from different prompts.
  A W that moves nothing gives INCONCLUSIVE, never PASS. The real `hf-ref` runs on a tiny CPU model (`--torch-gdn`):
  the patch at W = 0 is `torch.equal` to the library logits and a random W moves them.
* Re-run:

      TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python test_negeig_cpu.py --src <unpacked sglang pkg dir> \
          --hf-patch ../../negeig_retrofit/patch.py --tinyqwen <tiny qwen3_5 dir>
      CUDA_VISIBLE_DEVICES= python check_parity.py selftest --tinyqwen <tiny qwen3_5 dir>

Caveat on the CPU interpreter: Triton 3.8's CPU interpreter rounds fp32 to bf16 toward zero, while GPUs and torch
round to nearest. The CPU tests therefore claim bit identity only for the torch path and compare the interpreter
kernels to a round-toward-zero twin. Bit identity of the real kernels at W = 0 is what G1 measures on the box.

## What only the GPU box can verify

* G1 to G3 above, with real kernels, real bf16 rounding, and the real 27B checkpoint.
* The custom op and both Triton kernels under CUDA graph capture and replay (decode `full`, and `tc_piecewise`
  prefill graphs). The eager-first run in the launch section separates graph failures from math failures.
* `tl.sigmoid` on device against the torch sigmoid that the stock path and the HF reference use.
* Tensor parallel above 1 (row sharding is unit-tested, the collective layout is not). Pipeline parallel is untested.
* Throughput cost of the extra fp32 matmul plus a custom op per GDN layer. No benchmark was run here.
* Weight sync from the trainer into a live server (fp32 W through verl).

## Open risks

* The custom op under `tc_piecewise` or `full` CUDA graphs is unproven until G1 passes with graphs on. Dynamo
  fullgraph tracing is covered on CPU with the eager backend. CUDA graph capture and inductor are box-only.
* The stock G1 baseline must be launched with the same flags as the patched server: the three `--linear-attn-*`
  triton flags, `--disable-radix-cache`, the same cuda-graph config. On SM100+ SGLang picks the FlashInfer decode
  backend when the flag is unset (bf16 mamba-ssm-dtype), so a stock server without the explicit flags runs a
  different kernel and G1 would compare two kernels instead of two servers.
* `--target-t 0.5` perturbs all 48 gated layers at once and may scramble the model, which amplifies bf16 noise
  and can fail G2 for a reason that is not a bug. If the effect is large and G2 fails the noise check, retry with
  `--target-t 0.15`, using the same gates file for the server and for `hf-ref`.
* G1 bit identity depends on the real `tl.sigmoid` and round-to-nearest matching the stock kernel. If G1 fails by
  one bf16 ulp in a few positions, read the first differing layer before loosening anything.
* Stock run-to-run repeatability is an input: if G0 fails, G1 is INCONCLUSIVE, by design.
* Decode-vs-prefill agreement (G3) can differ by a bf16 ulp from different fp32 matmul shapes and kernels (the
  producer runs in row chunks of 8192, about 1.5e-6 fp32 noise against a single HF call).
* FlashInfer and CuTe prefill with beta above 1 are not verified and are refused by default.
* Target verify and speculative decoding, replay-SSM, and the generic (non-packed) decode are refused for gated layers.
* A TF32 setting (`--enable-tf32-matmul`) is overridden around the producer only. Anything else in the model that
  consumes the gate is unaffected.
* The checkpoint-load flow under overlap or capture-safe loading was not exercised. `--load-format dummy` is
  unsupported.
* `check_parity.py collect` assumes `--disable-radix-cache` and that the server returns full input logprobs from
  position 0. It stops with an error on a length mismatch instead of comparing short lists.
* The LoRA merge is bf16: the `delta_kept` gate protects the adapter, but the merged model still needs its own eval
  against the HF-side LoRA model.
