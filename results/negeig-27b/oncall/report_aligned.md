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
| untouched | 64 | 95.0 | 100.0 | 96.8 | 92.6 | 93.8 | 570 / 2235 | 195 / 194 | 2429 | 61759 | 65887 | 0.42 | 4.63 | 5.39 | 0 | 1 | 0 | 45 / 5956 |
| ctrl_s0 | 64 | 95.8 | 95.8 | 95.3 | 96.4 | 95.7 | 182 / 883 | 82 / 110 | 993 | 34050 | 36392 | 1.09 | 1.90 | 2.24 | 0 | 3 | 0 | 45 / 5956 |
| wide_s0 | 64 | 90.0 | 96.9 | 88.5 | 92.4 | 86.8 | 545 / 2597 | 68 / 81 | 2679 | 58740 | 59830 | 0.35 | 3.14 | 3.68 | 0 | 2 | 0 | 45 / 5956 |

Accuracy % by request kind (reasoning tokens per reached turn, mean):

| arm | ack_open | page_role | page_sev | resolve_mitigated | rollback | rollback_if |
|---|---|---|---|---|---|---|
| untouched | 98.5 (1854, n 67) | 96.5 (1438, n 77) | 93.2 (2469, n 59) | 95.3 (2112, n 43) | 93.1 (3012, n 51) | 90.7 (3137, n 43) |
| ctrl_s0 | 99.5 (370, n 67) | 92.2 (1276, n 77) | 98.6 (592, n 59) | 99.2 (286, n 43) | 90.5 (1736, n 51) | 95.3 (961, n 43) |
| wide_s0 | 94.7 (1910, n 67) | 92.0 (2478, n 77) | 84.5 (4232, n 59) | 86.9 (2686, n 43) | 90.2 (2190, n 51) | 89.5 (2034, n 43) |

Accuracy % by position (events delivered before the turn):

| arm | 1-64 | 65-128 | 129-256 | 257-512 |
|---|---|---|---|---|
| untouched | 100.0 (n 128) | 98.2 (n 93) | 88.1 (n 113) | 66.7 (n 6) |
| ctrl_s0 | 97.5 (n 128) | 96.8 (n 93) | 93.5 (n 113) | 86.1 (n 6) |
| wide_s0 | 97.7 (n 128) | 90.6 (n 93) | 83.0 (n 113) | 50.0 (n 6) |

| arm | call-turn acc % | calls <= 4 % | calls >= 5 % | action no-call acc % | events-turn no-call % | all turns % | exact sessions % | events-turn reasoning mean |
|---|---|---|---|---|---|---|---|---|
| untouched | 94.7 | 95.0 | 83.3 | 95.5 | 93.3 | 93.5 | 73.7 | 2847 |
| ctrl_s0 | 94.6 | 94.5 | 100.0 | 98.2 | 95.8 | 96.0 | 78.9 | 1672 |
| wide_s0 | 87.4 | 88.5 | 45.2 | 95.5 | 95.3 | 94.3 | 73.7 | 2572 |

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
| oncall | think_on | wide_s0 vs untouched | 64 | accuracy (primary turns) | -4.96 points | [-8.88, -0.98] |
| oncall | think_on | wide_s0 vs untouched | 64 | reasoning tokens per turn | 362.16 tokens | [-272.93, 984.42] |
| oncall | think_on | wide_s0 vs untouched | 64 | total tokens per turn | 249.47 tokens | [-384.73, 867.40] |
| oncall | think_on | wide_s0 vs untouched | 64 | reasoning per turn, both correct | -87.33 tokens | [-429.88, 235.94] |
| oncall | think_on | wide_s0 vs untouched | 64 | reasoning tokens per task | -3018.78 tokens | [-14036.60, 7253.24] |
| oncall | think_on | wide_s0 vs untouched | 64 | total tokens per task | -6056.95 tokens | [-17072.42, 4137.52] |
| oncall | think_on | wide_s0 vs untouched | 64 | accuracy per 1k reasoning | -0.08 per 1k | [-0.21, 0.04] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | accuracy (primary turns) | 0.83 points | [-2.21, 3.81] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | reasoning tokens per turn | -1352.60 tokens | [-1916.81, -788.48] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | total tokens per turn | -1436.39 tokens | [-1996.58, -873.43] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | reasoning per turn, both correct | -982.20 tokens | [-1265.05, -709.20] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | reasoning tokens per task | -27709.03 tokens | [-41721.24, -14541.04] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | total tokens per task | -29494.94 tokens | [-43440.91, -16405.79] |
| oncall | think_on | ctrl_s0 vs untouched | 64 | accuracy per 1k reasoning | 0.66 per 1k | [0.34, 1.13] |
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

- wide_s0 oncall think_off: aligned, 0 sessions cut to the turns every arm reached, 0 sessions dropped
- ctrl_s0 oncall think_off: aligned, 0 sessions cut to the turns every arm reached, 0 sessions dropped
- untouched oncall think_off: aligned, 0 sessions cut to the turns every arm reached, 0 sessions dropped
- wide_s0 oncall think_on: 41 sessions in flight, scored on their closed turns (5956 turns in flight)
- wide_s0 oncall think_on: aligned, 45 sessions cut to the turns every arm reached, 0 sessions dropped
- ctrl_s0 oncall think_on: 41 sessions in flight, scored on their closed turns (5956 turns in flight)
- ctrl_s0 oncall think_on: aligned, 45 sessions cut to the turns every arm reached, 0 sessions dropped
- untouched oncall think_on: 32 sessions in flight, scored on their closed turns (5956 turns in flight)
- untouched oncall think_on: aligned, 45 sessions cut to the turns every arm reached, 0 sessions dropped
- wide_s0 s1gen think_off: aligned, 0 sessions cut to the turns every arm reached, 0 sessions dropped
- ctrl_s0 s1gen think_off: aligned, 0 sessions cut to the turns every arm reached, 0 sessions dropped
- untouched s1gen think_off: aligned, 0 sessions cut to the turns every arm reached, 0 sessions dropped
