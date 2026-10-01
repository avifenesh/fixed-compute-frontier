# S1 state-tracking data: build manifest

Built 2026-09-30 from `experiments/negeig_27b/data` (`build.py --split train --n 5000`, `--split eval --n 50`), Qwen3.8-27B tokenizer (revision 1d4bf0f2) through its chat template with thinking off. Every answer in every session was recomputed by the domain's independent checker from the rendered text. Output lives in `/data/ai-ml/models/_runs/negeig-27b/data/s1` (regenerable: generation is deterministic in the seeds).

| file | sessions | turns (mean) | tokens mean / max (200 sampled) | sha256 |
|---|---|---|---|---|
| custody.train.jsonl | 5000 | 17.3 | 2267 / 6419 | `8c748f7ed34c5a80` |
| toggles.train.jsonl | 5000 | 17.4 | 2468 / 6202 | `1cf2da7d90a4614d` |
| fsys.train.jsonl | 5000 | 17.5 | 2525 / 4754 | `5b0bfcd61a2b5b97` |
| codetrace.train.jsonl | 5000 | 17.2 | 1775 / 3982 | `f6eef6dcf336691e` |
| ops.train.jsonl | 5000 | 17.4 | 2936 / 7332 | `a1e84bb8ca41e508` |
| orders.train.jsonl | 5000 | 17.6 | 3462 / 9223 | `98088e6be523f2c3` |
| custody.eval.jsonl | 200 | 54.8 | 7665 / 25591 | `b4656e53d40adb77` |
| toggles.eval.jsonl | 200 | 55.1 | 7745 / 23300 | `a85528504675dd02` |
| fsys.eval.jsonl | 200 | 55.1 | 7270 / 17352 | `cb06d4491048f645` |
| codetrace.eval.jsonl | 200 | 54.7 | 5570 / 15690 | `2fa7abd919f17365` |
| ops.eval.jsonl | 200 | 55.0 | 8961 / 26914 | `55774207406833cc` |
| orders.eval.jsonl | 200 | 55.0 | 10966 / 34094 | `d89f5b61d0fc9e61` |
