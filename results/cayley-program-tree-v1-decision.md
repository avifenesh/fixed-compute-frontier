# Full-budget Cayley program tree v1 decision

## Decision: exact v1 closed before language training

The frozen candidate fit on the G7 GPU, but its first forward pass was
non-finite.  This was not an OOM and not a BF16-only numerical issue.

The smoke reserved 49,383,735,296 bytes on a 101,975,851,008-byte device and
completed in 2.49 seconds, but language loss and router auxiliary were NaN.
The preregistered follow-up traced the same frozen model in BF16 and FP32.

The first decoder layer returned a finite but heavy-tailed MLP output.  The
second tree therefore received an RMS-normalized input with RMS 0.9999 and
maximum coordinate 19.39.  Within that tree, the recurrence amplified the
outlier as follows:

| Depth | projected RMS | projected max | activation RMS | hidden RMS | hidden max |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 19.19 | 4.64 | 1.01 | 40.50 |
| 1 | 1.014 | 39.83 | 7.85 | 1.17 | 178.99 |
| 2 | 1.172 | 173.70 | 72.69 | 5.62 | 2,293.66 |
| 4 | 17.67 | 4,327.96 | 39,693.87 | 149.43 | 35,001.20 |
| 6 | 23,764.15 | 9.74e6 | 2.27e10 | 9.17e8 | 3.93e11 |
| 8 | 1.92e17 | 6.59e19 | 4.17e35 | 2.52e33 | 9.54e35 |

At depth 8, the activation and then the state first became non-finite.  FP32
reproduced the same trajectory and failure.  The cause is structural: applying
a quadratic gated update recursively gives an effective polynomial degree that
doubles with depth.  RMS normalization outside the FFN cannot bound the hidden
trajectory inside it.

Changing the payload initialization would only move the explosion threshold.
The exact unbounded recurrence is therefore closed.  Any successor must prove
a depth-independent stability bound before another language run.

## Evidence integrity

- Smoke result SHA-256:
  `d406e2524b9d6a1d30df77713142f7fdc10c382a765be15276ea13163aa9ed53`
- Smoke log SHA-256:
  `78d9cd143e250dd819b0dbbf4ff2920d743b21cb3f873fde446aaa8dbab4a51e`
- Diagnosis result SHA-256:
  `9da88b85e7d0fcb1be9b367bf3d6a461496d98203c2e92c5f4c7365148c15e56`
- Diagnosis source SHA-256:
  `1ca9b2ec3a56b81e81c31e24a823f752e7f7fea4519dcfa0aa54f26fad0af966`
- Diagnosis preregistration SHA-256:
  `ede2034e5229ecb7e5b3d50c6d440ee6960fbacef4c4a6d4cd198dd9c7251003`
