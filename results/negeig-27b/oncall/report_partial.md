# Generative probe report

Accuracy over primary turns (on-call: action turns; S1: every answer), unreached and truncated turns count 0. Tokens per reached primary turn (median / mean); per task = all reached turns of one item. reasoning = tokenizer count of reasoning_content; total = completion tokens.

## oncall, think_off

| arm | items | acc % | acc@64 | acc@256 | acc@1024 | acc@2048 | reasoning/turn med / mean | content/turn med / mean | total/turn mean | reasoning/task mean | total/task mean | acc per 1k reasoning | trunc gens % | trunc turns % | unreached | budget | ctx overflow items | in-flight items / turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untouched | 64 | 65.7 | 68.6 | 71.1 | 64.0 | 65.8 | 0 / 0 | 119 / 153 | 153 | 0 | 14450 | n/a | 0.02 | 0.03 | 0 | 161 | 0 | 0 / 0 |
| ctrl_s0 | 64 | 83.1 | 88.5 | 89.1 | 81.9 | 82.8 | 0 / 0 | 63 / 69 | 69 | 0 | 3886 | n/a | 0.00 | 0.00 | 0 | 8 | 0 | 0 / 0 |
| wide_s0 | 64 | 84.0 | 89.1 | 89.2 | 83.1 | 83.7 | 0 / 0 | 72 / 89 | 89 | 0 | 8801 | n/a | 0.00 | 0.00 | 0 | 3 | 0 | 0 / 0 |

Accuracy % by request kind (reasoning tokens per reached turn, mean):

| arm | ack_open | page_role | page_sev | resolve_mitigated | rollback | rollback_if |
|---|---|---|---|---|---|---|
| untouched | 74.5 (0, n 289) | 72.6 (0, n 393) | 59.4 (0, n 258) | 72.3 (0, n 240) | 47.8 (0, n 254) | 63.2 (0, n 262) |
| ctrl_s0 | 73.9 (0, n 289) | 96.2 (0, n 393) | 84.2 (0, n 258) | 94.1 (0, n 240) | 66.8 (0, n 254) | 78.2 (0, n 262) |
| wide_s0 | 73.4 (0, n 289) | 97.2 (0, n 393) | 84.8 (0, n 258) | 93.2 (0, n 240) | 70.4 (0, n 254) | 80.2 (0, n 262) |

Accuracy % by position (events delivered before the turn):

| arm | 1-64 | 65-128 | 129-256 | 257-512 | 513-1024 | 1025-2048 |
|---|---|---|---|---|---|---|
| untouched | 74.2 (n 128) | 61.7 (n 96) | 71.1 (n 192) | 64.6 (n 256) | 62.8 (n 512) | 65.8 (n 512) |
| ctrl_s0 | 93.2 (n 128) | 84.5 (n 96) | 87.3 (n 192) | 83.5 (n 256) | 78.2 (n 512) | 83.4 (n 512) |
| wide_s0 | 88.9 (n 128) | 86.1 (n 96) | 89.3 (n 192) | 83.1 (n 256) | 82.1 (n 512) | 82.9 (n 512) |

| arm | call-turn acc % | calls <= 4 % | calls >= 5 % | action no-call acc % | events-turn no-call % | all turns % | exact sessions % | events-turn reasoning mean |
|---|---|---|---|---|---|---|---|---|
| untouched | 62.2 | 62.1 | 65.0 | 71.8 | 98.0 | 90.6 | 12.5 | 0 |
| ctrl_s0 | 79.0 | 78.8 | 85.1 | 90.2 | 99.6 | 95.8 | 26.6 | 0 |
| wide_s0 | 80.4 | 80.1 | 88.7 | 90.3 | 98.8 | 95.4 | 25.0 | 0 |

## oncall, think_on

| arm | items | acc % | acc@64 | acc@256 | acc@1024 | acc@2048 | reasoning/turn med / mean | content/turn med / mean | total/turn mean | reasoning/task mean | total/task mean | acc per 1k reasoning | trunc gens % | trunc turns % | unreached | budget | ctx overflow items | in-flight items / turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untouched | 64 | 85.4 | 100.0 | 94.9 | 81.4 | 79.6 | 843 / 4347 | 197 / 240 | 4588 | 169888 | 178181 | 0.20 | 14.88 | 17.17 | 0 | 1 | 0 | 32 / 5223 |
| ctrl_s0 | 64 | 95.5 | 95.8 | 95.5 | 96.5 | 94.1 | 184 / 970 | 82 / 111 | 1081 | 39571 | 42228 | 0.98 | 2.08 | 2.45 | 0 | 3 | 0 | 41 / 5805 |
| wide_s0 | 64 | 89.7 | 96.9 | 88.3 | 92.0 | 87.0 | 595 / 2890 | 70 / 90 | 2980 | 74178 | 75486 | 0.31 | 3.58 | 4.19 | 0 | 3 | 0 | 41 / 5759 |

Accuracy % by request kind (reasoning tokens per reached turn, mean):

| arm | ack_open | page_role | page_sev | resolve_mitigated | rollback | rollback_if |
|---|---|---|---|---|---|---|
| untouched | 92.3 (3522, n 91) | 86.3 (3308, n 118) | 82.6 (4519, n 86) | 84.1 (4940, n 63) | 81.4 (5669, n 78) | 84.1 (4981, n 67) |
| ctrl_s0 | 98.1 (639, n 72) | 93.1 (1183, n 87) | 98.7 (549, n 65) | 99.3 (278, n 48) | 88.6 (2163, n 57) | 95.6 (925, n 45) |
| wide_s0 | 93.8 (2319, n 74) | 91.7 (2770, n 86) | 83.9 (4475, n 63) | 89.0 (2831, n 51) | 87.7 (2694, n 57) | 90.6 (2198, n 48) |

Accuracy % by position (events delivered before the turn):

| arm | 1-64 | 65-128 | 129-256 | 257-512 |
|---|---|---|---|---|
| untouched | 100.0 (n 128) | 98.3 (n 96) | 81.2 (n 192) | 59.0 (n 87) |
| ctrl_s0 | 97.5 (n 128) | 96.8 (n 93) | 92.4 (n 132) | 96.0 (n 21) |
| wide_s0 | 97.7 (n 128) | 90.8 (n 96) | 83.9 (n 141) | 67.9 (n 14) |

| arm | call-turn acc % | calls <= 4 % | calls >= 5 % | action no-call acc % | events-turn no-call % | all turns % | exact sessions % | events-turn reasoning mean |
|---|---|---|---|---|---|---|---|---|
| untouched | 85.6 | 85.9 | 75.0 | 85.0 | 80.2 | 81.6 | 46.9 | 5142 |
| ctrl_s0 | 94.1 | 94.0 | 100.0 | 98.3 | 95.8 | 95.9 | 69.6 | 1749 |
| wide_s0 | 87.1 | 88.1 | 53.1 | 95.1 | 94.7 | 93.8 | 60.9 | 2861 |

## s1gen, think_off

| arm | items | acc % | acc@256 | acc@1024 | reasoning/turn med / mean | content/turn med / mean | total/turn mean | reasoning/task mean | total/task mean | acc per 1k reasoning | trunc gens % | trunc turns % | unreached | budget | ctx overflow items | in-flight items / turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untouched | 192 | 57.7 | 61.6 | 56.7 | 0 / 0 | 3 / 9 | 9 | 0 | 666 | n/a | 0.25 | 0.25 | 0 | 0 | 0 | 0 / 0 |
| ctrl_s0 | 192 | 90.2 | 91.8 | 89.7 | 0 / 0 | 3 / 5 | 5 | 0 | 397 | n/a | 0.00 | 0.00 | 0 | 0 | 0 | 0 / 0 |
| wide_s0 | 192 | 90.2 | 92.0 | 89.7 | 0 / 0 | 3 / 5 | 5 | 0 | 395 | n/a | 0.00 | 0.00 | 0 | 0 | 0 | 0 / 0 |

Accuracy % by domain (reasoning tokens per reached turn, mean):

| arm | codetrace | custody | fsys | ops | orders | toggles |
|---|---|---|---|---|---|---|
| untouched | 32.9 (0, n 2411) | 50.3 (0, n 2430) | 62.9 (0, n 2476) | 71.7 (0, n 2430) | 54.3 (0, n 2468) | 73.9 (0, n 2471) |
| ctrl_s0 | 71.4 (0, n 2411) | 95.0 (0, n 2430) | 89.7 (0, n 2476) | 96.0 (0, n 2430) | 95.9 (0, n 2468) | 92.7 (0, n 2471) |
| wide_s0 | 71.2 (0, n 2411) | 95.1 (0, n 2430) | 89.8 (0, n 2476) | 96.4 (0, n 2430) | 95.6 (0, n 2468) | 92.6 (0, n 2471) |

Accuracy % by position (events delivered before the turn):

| arm | 0 | 1-64 | 65-128 | 129-256 | 257-512 | 513-1024 |
|---|---|---|---|---|---|---|
| untouched | 97.9 (n 192) | 63.8 (n 1393) | 56.9 (n 1425) | 59.5 (n 2959) | 57.3 (n 2820) | 54.5 (n 5897) |
| ctrl_s0 | 99.5 (n 192) | 94.7 (n 1393) | 92.3 (n 1425) | 89.2 (n 2959) | 90.0 (n 2820) | 88.8 (n 5897) |
| wide_s0 | 100.0 (n 192) | 95.8 (n 1393) | 92.6 (n 1425) | 89.4 (n 2959) | 89.5 (n 2820) | 88.6 (n 5897) |

| arm | strict % | answers after event <= 256 % | > 256 % | > 512 % |
|---|---|---|---|---|
| untouched | 57.7 | 61.1 | 55.4 | 54.5 |
| ctrl_s0 | 90.2 | 91.6 | 89.2 | 88.8 |
| wide_s0 | 90.2 | 92.0 | 88.9 | 88.6 |

## Paired comparisons (A minus B, 95% bootstrap interval over items)

| task | mode | A vs B | items | metric | diff | interval |
|---|---|---|---|---|---|---|
| oncall | think_off | wide_s0 vs ctrl_s0 | 64 | accuracy (primary turns) | 0.95 points | [-1.45, 3.10] |
| oncall | think_off | wide_s0 vs ctrl_s0 | 64 | reasoning tokens per turn | 0.00 tokens | [0.00, 0.00] |
| oncall | think_off | wide_s0 vs ctrl_s0 | 64 | total tokens per turn | 20.04 tokens | [11.22, 30.18] |
| oncall | think_off | wide_s0 vs ctrl_s0 | 64 | reasoning tokens per task | 0.00 tokens | [0.00, 0.00] |
| oncall | think_off | wide_s0 vs ctrl_s0 | 64 | total tokens per task | 4914.92 tokens | [3150.38, 6879.69] |
| oncall | think_off | wide_s0 vs untouched | 64 | accuracy (primary turns) | 18.33 points | [14.06, 23.41] |
| oncall | think_off | wide_s0 vs untouched | 64 | reasoning tokens per turn | 0.00 tokens | [0.00, 0.00] |
| oncall | think_off | wide_s0 vs untouched | 64 | total tokens per turn | -64.04 tokens | [-90.98, -37.89] |
| oncall | think_off | wide_s0 vs untouched | 64 | reasoning tokens per task | 0.00 tokens | [0.00, 0.00] |
| oncall | think_off | wide_s0 vs untouched | 64 | total tokens per task | -5648.94 tokens | [-8168.66, -3412.93] |
| oncall | think_off | ctrl_s0 vs untouched | 64 | accuracy (primary turns) | 17.38 points | [13.05, 22.56] |
| oncall | think_off | ctrl_s0 vs untouched | 64 | reasoning tokens per turn | 0.00 tokens | [0.00, 0.00] |
| oncall | think_off | ctrl_s0 vs untouched | 64 | total tokens per turn | -84.08 tokens | [-108.35, -60.38] |
| oncall | think_off | ctrl_s0 vs untouched | 64 | reasoning tokens per task | 0.00 tokens | [0.00, 0.00] |
| oncall | think_off | ctrl_s0 vs untouched | 64 | total tokens per task | -10563.86 tokens | [-13993.36, -7426.39] |
| oncall | think_on | wide_s0 vs ctrl_s0 | 64 | accuracy (primary turns) | -5.79 points | [-9.38, -2.19] |
| oncall | think_on | wide_s0 vs ctrl_s0 | 64 | reasoning tokens per turn | 1714.76 tokens | [1188.00, 2215.55] |
| oncall | think_on | wide_s0 vs ctrl_s0 | 64 | total tokens per turn | 1685.86 tokens | [1159.39, 2187.23] |
| oncall | think_on | wide_s0 vs ctrl_s0 | 64 | reasoning per turn, both correct | 931.91 tokens | [665.98, 1199.67] |
| oncall | think_on | wide_s0 vs ctrl_s0 | 64 | reasoning tokens per task | 24690.25 tokens | [15512.48, 33794.51] |
| oncall | think_on | wide_s0 vs ctrl_s0 | 64 | total tokens per task | 23437.98 tokens | [14400.39, 32342.43] |
| oncall | think_on | wide_s0 vs ctrl_s0 | 64 | accuracy per 1k reasoning | -0.74 per 1k | [-1.19, -0.45] |
| oncall | think_on | wide_s0 vs untouched | 64 | accuracy (primary turns) | -4.06 points | [-8.26, 0.26] |
| oncall | think_on | wide_s0 vs untouched | 64 | reasoning tokens per turn | 329.79 tokens | [-362.86, 1015.64] |
| oncall | think_on | wide_s0 vs untouched | 64 | total tokens per turn | 224.22 tokens | [-465.28, 908.85] |
| oncall | think_on | wide_s0 vs untouched | 64 | reasoning per turn, both correct | -13.58 tokens | [-463.57, 406.69] |
| oncall | think_on | wide_s0 vs untouched | 64 | reasoning tokens per task | -7489.84 tokens | [-22276.28, 6621.39] |
| oncall | think_on | wide_s0 vs untouched | 64 | total tokens per task | -11555.50 tokens | [-26808.92, 2994.13] |
| oncall | think_on | wide_s0 vs untouched | 64 | accuracy per 1k reasoning | -0.06 per 1k | [-0.17, 0.04] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | accuracy (primary turns) | 3.14 points | [-0.50, 6.79] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | reasoning tokens per turn | -1926.25 tokens | [-2604.75, -1234.64] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | total tokens per turn | -2004.89 tokens | [-2669.89, -1320.45] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | reasoning per turn, both correct | -1240.66 tokens | [-1588.53, -901.00] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | reasoning tokens per task | -44261.38 tokens | [-63833.56, -25656.64] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | total tokens per task | -46180.67 tokens | [-65598.90, -27596.58] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | accuracy per 1k reasoning | 0.66 per 1k | [0.39, 1.05] |
| s1gen | think_off | wide_s0 vs ctrl_s0 | 192 | accuracy (primary turns) | 0.01 points | [-0.48, 0.50] |
| s1gen | think_off | wide_s0 vs ctrl_s0 | 192 | reasoning tokens per turn | 0.00 tokens | [0.00, 0.00] |
| s1gen | think_off | wide_s0 vs ctrl_s0 | 192 | total tokens per turn | -0.02 tokens | [-0.04, -0.01] |
| s1gen | think_off | wide_s0 vs ctrl_s0 | 192 | reasoning tokens per task | 0.00 tokens | [0.00, 0.00] |
| s1gen | think_off | wide_s0 vs ctrl_s0 | 192 | total tokens per task | -1.69 tokens | [-2.95, -0.53] |
| s1gen | think_off | wide_s0 vs untouched | 192 | accuracy (primary turns) | 32.41 points | [29.88, 35.13] |
| s1gen | think_off | wide_s0 vs untouched | 192 | reasoning tokens per turn | 0.00 tokens | [0.00, 0.00] |
| s1gen | think_off | wide_s0 vs untouched | 192 | total tokens per turn | -3.55 tokens | [-13.02, 1.14] |
| s1gen | think_off | wide_s0 vs untouched | 192 | reasoning tokens per task | 0.00 tokens | [0.00, 0.00] |
| s1gen | think_off | wide_s0 vs untouched | 192 | total tokens per task | -271.57 tokens | [-961.20, 88.17] |
| s1gen | think_off | ctrl_s0 vs untouched | 192 | accuracy (primary turns) | 32.41 points | [29.88, 35.13] |
| s1gen | think_off | ctrl_s0 vs untouched | 192 | reasoning tokens per turn | 0.00 tokens | [0.00, 0.00] |
| s1gen | think_off | ctrl_s0 vs untouched | 192 | total tokens per turn | -3.53 tokens | [-12.99, 1.17] |
| s1gen | think_off | ctrl_s0 vs untouched | 192 | reasoning tokens per task | 0.00 tokens | [0.00, 0.00] |
| s1gen | think_off | ctrl_s0 vs untouched | 192 | total tokens per task | -269.88 tokens | [-959.64, 90.07] |

## Notes

- wide_s0 oncall think_on: 41 sessions in flight, scored on their closed turns (5759 turns in flight)
- ctrl_s0 oncall think_on: 41 sessions in flight, scored on their closed turns (5805 turns in flight)
- untouched oncall think_on: 32 sessions in flight, scored on their closed turns (5223 turns in flight)
