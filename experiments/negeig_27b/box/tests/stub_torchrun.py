#!/usr/bin/env python3
"""CPU stand-in for `torchrun ... train27.py ...` (box tests only). It emulates what the box scripts read from the
trainer: log.jsonl events (config, mem_probe, step, eval), profile_rank0.json, ckpt_NNNNNN.pt (newest --keep_ckpts
kept), lengths_*.json, complete.json, and the exit code. No torch, no GPU, no sleeping except the `sleep` action.

Per-arm plan: $STUB_PLAN_DIR/<arm>.plan, one action per line, consumed one per launch (a count file keeps the place).
An action ending in * repeats forever once reached. Actions:
  ok           run to the end from the newest checkpoint and exit 0 (the default when the plan is empty or absent)
  fail         exit 1 at once
  ckfail:N     advance to step N writing checkpoints, then exit 1
  sleep        wait until SIGTERM, then exit 143
Every launch appends one JSON line to $STUB_LOG (arm, nproc, port, CUDA_VISIBLE_DEVICES, NCCL_NVLS_ENABLE, argv, action).
Synthetic throughput: STUB_RATE_PER_GPU tok/s per rank (default 4000) times STUB_SCALE_8 (default 0.9) at 8 ranks.
Synthetic memory: peak = 60 + tokens/1000 * slope, slope 3 with --grad_ckpt 1 and STUB_OFF_SLOPE (default 12) with 0.
STUB_OOM_GC0=<pool|1>: with --grad_ckpt 0 and --mem_probe, die with a CUDA out-of-memory at that pool (1 = the first pool).

NCCL probe mode (the script is nccl_probe.py): prints one NCCL_RESULT line per size like accept.sh's probe. Knobs:
  STUB_NCCL_MS0 / STUB_NCCL_MS1   ms per all-reduce with NCCL_NVLS_ENABLE=0 / 1 (default 10 / 10)
  STUB_NCCL_FAIL1=1               exit 1 with NCCL_NVLS_ENABLE=1      STUB_NCCL_HANG1=1   hang under NVLS=1 (until SIGTERM)
  STUB_NCCL_FAIL0=1               exit 1 with NCCL_NVLS_ENABLE=0
  STUB_NCCL_BUSBW                 bus bandwidth printed (default 400); below MIN_NCCL_GBPS the probe exits 1, like the real one
  STUB_NCCL_MS0_SEQ / _MS1_SEQ    comma list of ms, one per call of that NVLS value (the last repeats); beats MS0/MS1
Each probe launch appends {"script": "nccl_probe.py", "nvls", "nproc", "argv"} to $STUB_LOG.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

argv = sys.argv[1:]
nproc, port, i = 1, 0, 0
while i < len(argv) and argv[i].startswith("--") and argv[i] != "--":
    if argv[i] == "--nproc_per_node":
        nproc = int(argv[i + 1]); i += 2
    elif argv[i] == "--master_port":
        port = int(argv[i + 1]); i += 2
    elif argv[i] == "--standalone":
        i += 1
    else:
        i += 2
script, targs = argv[i], argv[i + 1:]

if os.path.basename(script) == "nccl_probe.py":
    nvls = os.environ.get("NCCL_NVLS_ENABLE", "unset")
    if os.environ.get("STUB_LOG"):
        with open(os.environ["STUB_LOG"], "a") as f:
            f.write(json.dumps({"arm": "-", "script": "nccl_probe.py", "nvls": nvls, "nproc": nproc, "argv": targs}) + "\n")
    if nvls == "1" and os.environ.get("STUB_NCCL_HANG1") == "1":
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        time.sleep(600)
        sys.exit(0)
    if (nvls == "1" and os.environ.get("STUB_NCCL_FAIL1") == "1") or (nvls == "0" and os.environ.get("STUB_NCCL_FAIL0") == "1"):
        print("stub nccl: planned failure with NCCL_NVLS_ENABLE=" + nvls, file=sys.stderr)
        sys.exit(1)
    key = "STUB_NCCL_MS" + ("1" if nvls == "1" else "0")
    ms = float(os.environ.get(key, "10"))
    seq = [x for x in os.environ.get(key + "_SEQ", "").split(",") if x]
    if seq:
        cf = Path(os.environ.get("STUB_PLAN_DIR", "/tmp")) / f"nccl{nvls}.count"
        n = int(cf.read_text()) if cf.exists() else 0
        cf.write_text(str(n + 1))
        ms = float(seq[min(n, len(seq) - 1)])
    busbw = float(os.environ.get("STUB_NCCL_BUSBW", "400"))
    for mib in [int(x) for x in targs] or [512]:
        print(f"NCCL_RESULT mib={mib} ms={ms:.3f} busbw_gbs={busbw:.1f} world={nproc} nvls={nvls}")
    floor = float(os.environ.get("MIN_NCCL_GBPS", "0"))
    if busbw < floor:
        print(f"NCCL_TOO_SLOW: {busbw:.0f} GB/s is below MIN_NCCL_GBPS {floor:.0f}")
        sys.exit(1)
    sys.exit(0)

ap = argparse.ArgumentParser()
for name, typ, default in [("arm", str, "wide"), ("out", str, ""), ("seed", int, 0), ("steps", int, 700),
                           ("eval_every", int, 50), ("save_every", int, 50), ("recall_every", int, 200),
                           ("keep_ckpts", int, 4), ("grad_ckpt", int, 1), ("mem_probe", int, 0),
                           ("profile_step", int, -1), ("max_steps_debug", int, 0), ("lr", float, 1e-4),
                           ("w_lr", float, 1e-4), ("model", str, ""), ("s1", str, ""), ("lm_replay", str, ""),
                           ("chat_replay", str, ""), ("tool_replay", str, "")]:
    ap.add_argument(f"--{name}", type=typ, default=default)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--lr_override", action="store_true")
a, _ = ap.parse_known_args(targs)

plan_dir = os.environ.get("STUB_PLAN_DIR", "")
action = "ok"
if plan_dir and (Path(plan_dir) / f"{a.arm}.plan").exists():
    lines = [x.strip() for x in (Path(plan_dir) / f"{a.arm}.plan").read_text().splitlines() if x.strip()]
    cnt_f = Path(plan_dir) / f"{a.arm}.count"
    n = int(cnt_f.read_text()) if cnt_f.exists() else 0
    cnt_f.write_text(str(n + 1))
    if lines:
        action = lines[min(n, len(lines) - 1)] if (n < len(lines) or lines[-1].endswith("*")) else "ok"
        action = action.rstrip("*")

rec = {"arm": a.arm, "nproc": nproc, "port": port, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
       "nvls": os.environ.get("NCCL_NVLS_ENABLE"), "action": action, "argv": targs, "script": script}
if os.environ.get("STUB_LOG"):
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(json.dumps(rec) + "\n")

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

if action == "fail":
    print("stub: planned failure", file=sys.stderr)
    sys.exit(1)
if action == "sleep":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    time.sleep(600)
    sys.exit(0)


def ckpts():
    return sorted(out.glob("ckpt_[0-9]*.pt"))


def save(step):
    (out / f"ckpt_{step:06d}.pt").write_bytes(b"x" * 4096)
    (out / f"trainable_{step:06d}.pt").write_bytes(b"y" * 512)
    for old in ckpts()[: -a.keep_ckpts]:
        old.unlink()


start = 0
if a.resume and ckpts():
    start = int(ckpts()[-1].name[5:11])
total = min(a.steps, a.max_steps_debug) if a.max_steps_debug else a.steps
stop_at = total
if action.startswith("ckfail:"):
    stop_at = min(total, int(action.split(":")[1]))

log = open(out / "log.jsonl", "a")


def emit(r):
    log.write(json.dumps(r) + "\n"); log.flush()


emit({"event": "config", "world": nproc, "args": vars(a)})
# length caches, as the real trainer writes them next to the run
for pool in ["s1"] + (["chat"] if a.chat_replay else []) + (["tool"] if a.tool_replay else []):
    p = out / f"lengths_{pool}_stub.json"
    if not p.exists():
        p.write_text(json.dumps([[100, 10]] * 4))

if a.mem_probe:
    slope = 3.0 if a.grad_ckpt else float(os.environ.get("STUB_OFF_SLOPE", "12"))
    pools = [("s1", 4096)] + ([("chat", 6000)] if a.chat_replay else []) + ([("tool", 12288)] if a.tool_replay else [])
    for name, tokens in pools:
        oom = os.environ.get("STUB_OOM_GC0", "")
        if not a.grad_ckpt and oom and (oom == "1" and name == pools[0][0] or oom == name):
            print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 GiB (stub, pool %s)" % name, file=sys.stderr)
            sys.exit(1)
        emit({"event": "mem_probe", "pool": name, "tokens": tokens, "grad_ckpt": a.grad_ckpt, "s": 1.5,
              "peak_gib": round(60 + tokens / 1000 * slope, 2)})
    if a.mem_probe == 2:
        sys.exit(0)

rate = float(os.environ.get("STUB_RATE_PER_GPU", "4000")) * nproc * (float(os.environ.get("STUB_SCALE_8", "0.9")) if nproc >= 8 else 1.0)
tok_step = 160000                         # the global batch: the same at every world size
if a.eval_every and start % a.eval_every == 0:
    emit({"event": "eval", "step": start})
t_train = t_all = 0.0
tokens = 0
peak0 = 60 + 4096 / 1000 * (3.0 if a.grad_ckpt else float(os.environ.get("STUB_OFF_SLOPE", "12")))
for step in range(start, stop_at):
    tokens += tok_step + (step % 7) * 100
    dt = (tok_step + (step % 7) * 100) / rate
    t_train += dt; t_all += dt
    if a.profile_step >= 0 and step == a.profile_step + 2:
        (out / "profile_rank0.json").write_text(json.dumps({"steps": [a.profile_step, a.profile_step + 2],
                                                            "device_share": {"gdn": 0.41, "attention": 0.12, "gemm": 0.36}}))
        t_train += 20.0; t_all += 20.0    # the table export lands in the step time
    if step % 10 == 0 or step == total - 1:
        batch = tok_step + (step % 7) * 100
        loads = [batch // nproc] * nproc
        loads[0] += batch - sum(loads)
        emit({"event": "step", "step": step + 1, "loss": {"s1": 1.0, "chat": 0.5}, "gnorm": 0.3, "lr": a.lr, "lr_w": a.w_lr,
              "tok_per_s": round(tokens / t_all, 1), "train_tok_per_s": round(tokens / t_train, 1), "elapsed_s": round(t_all, 1),
              "rank_compute_s": [round(2.0 * (1 + 0.03 * r), 2) for r in range(nproc)], "lpt_load_tokens": loads,
              "peak_mem_gib": round(peak0, 2), "w_norm": 0.1})
    if (a.save_every and (step + 1) % a.save_every == 0) or (step == total - 1):
        save(step + 1)
        t_all += 3.0
    if a.eval_every and (step + 1) % a.eval_every == 0:
        emit({"event": "eval", "step": step + 1}); t_all += 30.0
    if step + 1 == stop_at and stop_at < total:
        print("stub: planned crash at step", step + 1, file=sys.stderr)
        sys.exit(1)
(out / "complete.json").write_text(json.dumps({"steps": total}))
