# S4 general replay: build manifest

**Clean LM replay** (`experiments/negeig_retrofit/prep_replay_clean.py --code_frac 0.2 --he_frac 0.2`, Qwen3.8-27B
tokenizer 1d4bf0f2): FineWeb-Edu sample-10BT (English web), codeparrot-clean-valid (Python), FineWeb-2 heb_Hebr (Hebrew
web, ODC-BY). 512-token chunks, documents separated by EOS, test documents drawn before train (disjoint).
Train 62,500 chunks (32.0M tokens: 37,500 web, 12,500 code, 12,500 Hebrew); test 816 chunks (484 / 175 / 157).
`/data/ai-ml/models/_runs/negeig-27b/data/s4/replay_clean_q38.pt`, sha256 `49e59e78b1a1c43e`. WikiText is not used: its
tokenized format broke code generation in the 4B and 9B runs.

**Chat-replay prompts** (`experiments/negeig_27b/data/collect_prompts.py`): human-written only, open licences.
20,139 candidates (databricks-dolly-15k CC-BY-SA-3.0; OpenAssistant/oasst2 first user turns, Apache-2.0); dropped 264
duplicates, 792 outside 20 to 3,000 characters, 17 with an 8-word overlap with IFEval. Written: 6,002 (4,433 Dolly,
1,569 oasst2; 6,000 English, 2 Hebrew). `/data/ai-ml/models/_runs/negeig-27b/data/s4/chat_prompts.jsonl`, sha256 `4a063f28fe0e0897`.
oasst2 has almost no Hebrew, so Hebrew chat prompts are to be made by the untouched model translating a sample of these.
Responses come from the untouched Qwen3.8-27B (self-distillation); no hosted-model output is a prompt or a label.
