#!/usr/bin/env python3
"""Write the ledger G0 manifest for the negeig retrofit (validated by frontier_g0.py)."""

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

import tasks

SNAP = Path("/data/ai-ml/hf-models/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b")
REPO = Path(__file__).resolve().parents[2]
CKPT = ["config.json", "model.safetensors.index.json",
        "model.safetensors-00001-of-00002.safetensors", "model.safetensors-00002-of-00002.safetensors"]
TOK = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
N_LORA, N_W = 8_675_328, 1_966_080  # measured by the smoke run (fp32 adapters)
EVAL_LENGTHS, EVAL_N, EVAL_BATCH = (64, 256, 1024), 128, 8


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tree_sha(names):
    h = hashlib.sha256()
    total = 0
    for n in sorted(names):
        p = (SNAP / n).resolve()
        size = p.stat().st_size
        total += size
        h.update(f"{n}\0{size}\0{file_sha(p)}\n".encode())
    return h.hexdigest(), total


def main(out):
    ck_sha, ck_bytes = tree_sha(CKPT)
    tk_sha, tk_bytes = tree_sha(TOK)
    weights = sum((SNAP / n).resolve().stat().st_size for n in CKPT if n.endswith(".safetensors"))
    meta = ck_bytes - weights + tk_bytes
    tok = AutoTokenizer.from_pretrained(SNAP)
    V = tasks.Vocab(tok)
    trace = hashlib.sha256()
    cells = []
    for task in tasks.ALL_TASKS:
        for L in EVAL_LENGTHS:
            rng = random.Random(10_000 + 7919 * L + 104729 * tasks.ALL_TASKS.index(task))  # train.eval_tasks
            exs = [tasks.make(task, L, rng, V) for _ in range(EVAL_N)]
            for e in exs:
                trace.update(json.dumps([e.ids, e.label_pos, e.labels]).encode())
            cells.append({"name": f"{task}@{L}", "input_tokens": round(sum(len(e.ids) for e in exs) / EVAL_N),
                          "output_tokens": L, "concurrency": EVAL_BATCH})
    replay = Path("/data/ai-ml/models/_runs/negeig-retrofit/replay_wikitext103.pt")
    trace.update(file_sha(replay).encode())
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    if not freeze.strip():
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True).stdout
    pci = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().split(", ")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def art(aux):
        return {"model_reference": f"Qwen/Qwen3.5-4B-Base@{SNAP.name}",
                "checkpoint_sha256": ck_sha, "tokenizer_sha256": tk_sha,
                "learned_bytes": {"core": weights, "auxiliary": aux, "metadata": meta, "total": weights + aux + meta}}

    m = {
        "schema_version": 1,
        "experiment_id": "negeig-retrofit-qwen35-4b-base",
        "claim": {"scope": "lifecycle", "primary_edge": "E", "edges": ["E"],
                  "statement": "Retrofitting beta in (0,2) into a pretrained Gated DeltaNet hybrid cuts tier-1 "
                               "state-tracking error in the 4x length window by at least 20% relative to the same "
                               "LoRA fine-tune with beta in (0,1), with no protected-slice regression.",
                  "preregistration": "results/negeig-retrofit-preregistration.md"},
        "artifacts": {
            "hash_scheme": "sha256(canonical path\\0bytes\\0file_sha256 records)",
            "note": "Both arms start from the same checkpoint; the arms differ in fp32 trainable adapters. "
                    "baseline = ctrl (LoRA only), candidate = wide (LoRA plus zero-init beta gate); at step 0 "
                    "they are bit-identical to the checkpoint (G0).",
            "baseline": art(4 * N_LORA),
            "candidate": art(4 * (N_LORA + N_W)),
        },
        "hardware": {
            "gpu_stratum": f"{pci[1]} {pci[2]}", "gpu_pci_id": pci[0], "driver": pci[3],
            "container_digest": "no-container;uv-venv=~/.venvs/negeig;pip-freeze-sha256="
                                + hashlib.sha256(freeze.encode()).hexdigest(),
            "server_commit": f"fixed-compute-frontier@{commit};transformers==5.17.0;flash-linear-attention==0.5.2;"
                             "peft==0.21.0;torch==2.14.0+cu130",
            "host": "owner local rig (laptop RTX 5090), authorized by the owner 2026-09-29",
        },
        "workload": {"trace_sha256": trace.hexdigest(), "frozen": True,
                     "note": "eval sets (5 tasks x 3 lengths x 128 sequences, fixed seeds) plus the replay chunk file",
                     "cells": cells},
        "quality": {
            "protected_slices": ["P1", "P2", "P3", "P4", "P5"],
            "noninferiority_margins": {"P1": 0.005, "P2": 0.02, "P3": 0.02, "P4": 0.02, "P5": 0.02},
            "slice_definitions": {
                "P1": "held-out WikiText-103 NLL, relative increase over ctrl (upper 95% bound)",
                "P2": "tier-1 accuracy, 1x window, absolute drop vs ctrl",
                "P3": "tier-2 accuracy, 1x window, absolute drop vs ctrl",
                "P4": "each tier-1 task, 4x window, absolute drop vs ctrl",
                "P5": "held-out NLL, relative increase over the untouched base (upper 95% bound)"},
        },
        "planned_training_seeds": [0, 1, 2, 3, 4],
    }
    Path(out).write_text(json.dumps(m, indent=1) + "\n")
    print(out, ck_sha, tk_sha, trace.hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
