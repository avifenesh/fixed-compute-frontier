# Phase-elastic v3: fresh-seed 50M packed phase-faithful replication

## Purpose

Seed 6673 showed a 0.5662% mixed-mode suffix gain at 10M tokens.  A deterministic
recreation later differed from the historical run by up to `6.72e-4` NLL, so
its preregistered `1e-7` historical-recreation gate correctly failed.  Within
that new run, however, fake-quant versus independently reloaded packed weights
matched exactly (`0.0` maximum bucket loss difference), retained a 0.5658%
gain, used exactly 2,338,816 persistent FFN bytes per layer, and contained no
training shadows.

This experiment avoids historical bitwise reproduction.  It trains once at a
fresh seed for 50M tokens, compares pre-export and post-reload weights inside
the same run, and judges only the independently reloaded packed model.

## Frozen experiment

- fresh seed 9901;
- arms in order: BF16 dense baseline, phase-faithful v3;
- `D=384,M=1024,L=12`, sequence 512, microbatch 32, accumulation 2;
- 1,525 AdamW steps = 49,971,200 loaded prediction tokens per arm;
- peak LR `3e-4`, 100-step warmup, cosine decay, betas `(0.9,0.95)`, weight
  decay 0.1, clip 1.0;
- the same seven training boundaries and five evaluation boundaries as v3;
- ordinary full-sequence LM loss for both arms;
- 128 paired one-pass validation batches;
- actual BF16 cached teacher-forced evaluation at `P={256,480}`, 16 batches of
  8 sequences per boundary.  Paired intervals use 128 per-sequence differences
  and Student's `t(127)=1.978819534`, not eight batch means;
- H100, Torch `2.5.1+cu124`, CUDA 12.4, Transformers 4.57.6.

## Frozen artifact path

After candidate training:

1. evaluate the fake-quant phase grid;
2. replace every FFN training weight with persistent packed INT8+ternary or W4
   buffers;
3. serialize, delete the source model, construct an independent packed
   skeleton, and strictly reload;
4. perform all decision evaluations on the reloaded packed model.

The unpacking oracle proves discrete quality and bytes, not latency.

## Frozen pass

The reloaded packed candidate advances only if:

1. its aggregate mixed suffix NLL beats dense by at least 0.25%, with a wholly
   favorable paired 95% interval;
2. it retains at least half of the seed-6673 relative gain and at least 0.10%
   absolute gain;
3. every evaluated boundary improves over dense;
4. the optional branch improves mixed suffix NLL over the same packed model's
   base-only path by at least 0.10%, with a favorable paired interval, in both
   one-pass and actual cached execution;
5. no populated first-token or decode-offset bucket regresses by more than
   0.10% versus either dense or the candidate's base-only path, in both one-pass
   and actual cached execution;
6. first-token NLL is noninferior within 0.05%, including the paired upper
   bound;
7. the base-only full-sequence endpoint is noninferior within 0.05%, and the
   full endpoint beats dense by at least 0.25%;
8. aggregate actual BF16 cached mixed-mode suffix NLL beats cached dense by at
   least 0.10%, with a favorable interval, and cached first-token NLL is
   noninferior within 0.05%;
9. every post-reload bucket equals its in-run pre-export value within `1e-7`;
10. strict reload, exact bytes, no shadows, code/scale bounds, cache semantics,
   finiteness, data, and protocol checks pass.

The final source and this preregistration are pinned by an external integrity
manifest before execution.

A pass establishes a replicated, exact-storage quality candidate.  Equal or
lower serving cost still requires a physical packed H100 kernel and end-to-end
phase latency measurement.
