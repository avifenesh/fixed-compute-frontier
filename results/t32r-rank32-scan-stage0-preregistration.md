# T32R rank-32 boundary scan — CPU preregistration

Status: **FROZEN BEFORE IMPLEMENTATION**  
Date: 2026-07-31

Implement exactly `results/t32r-rank32-scan-algebra.md` in NumPy with seed
`32032`, `d=384`, `r=32`, four heads, two 128-position documents, and float32
scan arithmetic.

Mandatory worlds:

1. random finite query, embedding table, records, and parameters;
2. all-zero inputs;
3. no-handle path returns the input boundary state bit-exactly;
4. explicit `R_A=I,R_B=-I` role witness changes sign under record swap;
5. controlled score gaps `Delta in {2,4,8,12,16}` over 128 positions, checking
   the exact target softmax weight and the registered error bound;
6. forward repeat and reversed document construction order;
7. exact parameter and multiplication ledgers from the algebra.

Require all arrays and outputs finite, repeat outputs bit-identical, independent
reference maximum absolute error at most `1e-6`, every observed concentration
error within its theorem bound plus `1e-6`, total entries 943,538 within
967,680, and total scan multiplies exactly 3,230,144 within 3,700,000.

The source may not read corpus/evaluator files, model weights, or CUDA. A pass
admits only a frozen H100 block-latency microbenchmark and learnability
preflight. A fail closes this exact scan without changing rank, heads, filter,
or tolerance.
