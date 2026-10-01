# S2 agentic state-tracking episodes: build manifest

Built 2026-10-01 from `experiments/negeig_27b/data/s2.py` in the Hebrew lane's `agentic_env` item format (canonical call keys identical to `agentic_env.call_key`, checked). Each episode: a log tool returning an S1 simulator run (32 to 128 events), and a request whose correct actions depend on the state at the end of the log. Every episode's expected actions were re-derived from the log and request text through the domain's S1 `replay()` checker; a red test caught 646 of 646 corrupted expectations. Output in `/data/ai-ml/models/_runs/negeig-27b/data/s2`.

| file | episodes | actions per episode (5 = 5 or more) | sha256 |
|---|---|---|---|
| s2_ops.train.jsonl | 1500 | {0: 436, 1: 729, 2: 137, 3: 95, 4: 50, 5: 53} | `772079a9e2b04c62` |
| s2_toggles.train.jsonl | 1500 | {0: 41, 1: 653, 2: 265, 3: 235, 4: 163, 5: 143} | `27675cbc98f8ec88` |
| s2_custody.train.jsonl | 1500 | {1: 494, 2: 499, 3: 507} | `191d1c8446dec498` |
| s2_orders.train.jsonl | 1500 | {0: 231, 1: 170, 2: 212, 3: 234, 4: 206, 5: 447} | `2d37a7a4f1611846` |
| s2_ops.eval.jsonl | 150 | {0: 55, 1: 56, 2: 16, 3: 8, 4: 5, 5: 10} | `e113604431bc6360` |
| s2_toggles.eval.jsonl | 150 | {0: 3, 1: 66, 2: 19, 3: 20, 4: 22, 5: 20} | `c029d64964753531` |
| s2_custody.eval.jsonl | 150 | {1: 53, 2: 52, 3: 45} | `4f62e4737a1f1e3e` |
| s2_orders.eval.jsonl | 150 | {0: 39, 1: 18, 2: 14, 3: 20, 4: 13, 5: 46} | `0c5cb9d201301762` |
