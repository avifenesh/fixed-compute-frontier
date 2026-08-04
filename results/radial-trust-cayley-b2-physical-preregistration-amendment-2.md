# Radial-trust Cayley B2 physical preregistration — Amendment 2

Date: 2026-08-01  
Status: **PROPOSED NORMATIVE AMENDMENT; DO NOT IMPLEMENT OR RUN**

## 1. Parent and scope

This document is a prospective normative amendment to the effective B2 paper:

- `radial-trust-cayley-b2-physical-preregistration.md`, frozen SHA-256
  `31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`;
- `radial-trust-cayley-b2-physical-preregistration-amendment-1.md`, frozen
  SHA-256
  `9259dc8c5c4e0bb3f3ac0bb6bda207f87b8d840dc93bd36b7e562b0d7affa900`.

It changes only the previously unspecified call protocol and canonical order
for `nvmlDeviceGetSupportedMemoryClocks` and the corresponding
`nvmlDeviceGetSupportedGraphicsClocks` records in parent Section 7.0. It does
not alter the process-list retry protocol in parent lines 631--642, any
architecture, numerical operator, artifact, workload, control, statistical
gate, timing rule, byte ledger, or device-acquisition rule.

This amendment is not effective until it passes an independent paper audit.
The effective preregistration identity will then be the ordered hash triple
`(parent_sha256, amendment_1_sha256, amendment_2_sha256)`. The historical
parent and Amendment 1 remain byte-for-byte unchanged.

## 2. Defect

The parent requires every supported memory clock and its corresponding
supported graphics clocks, but does not define the variable-length NVML call
protocol. The nearby maximum-three-attempt rule is explicitly scoped to the
three boundary process-list APIs and cannot silently govern supported-clock
queries.

The parent also requires every integer to be a decimal string and arrays to be
sorted bytewise by their canonical path or key. A numeric sort of clock MHz
values is different from a bytewise sort of their decimal-string keys; for
example, bytewise order is `"1000"` before `"900"`.

The pinned CUDA/NVML header
`/usr/local/cuda/include/nvml.h`, SHA-256
`28b51fbd44df16adf1e58229778414a4d1e7e05fdd4a74526ef0affb75f18416`,
defines `NVML_SUCCESS=0`, `NVML_ERROR_NOT_SUPPORTED=3`, and
`NVML_ERROR_INSUFFICIENT_SIZE=7`. Both supported-clock APIs accept an
`unsigned int *count` and `unsigned int *clocksMHz`; their documented
insufficient-size result writes the required element count.

## 3. Normative call protocol

For each supported-clock list, perform exactly one fail-closed two-call query;
there is no retry, alternate API, cached result, guessed capacity, or fallback.
Both calls use the identical full-GPU device handle; for a graphics-clock list,
both calls also use the identical `memoryClockMHz` selector.
The complete transcript fields below are invariant fields. All integer inputs,
outputs, return codes, capacities, and clock values use canonical decimal
strings in the manifest; pointer states use JSON booleans.

### 3.1 First call

Initialize `count=0` and call the selected API with `clocksMHz=NULL`. Record
`first_input_count="0"`, `first_pointer_null=true`, the exact first return code,
and the returned count.

Only these first-call outcomes are representable:

1. `NVML_ERROR_NOT_SUPPORTED` with returned `count=0`: record the exact return
   code and status `not_supported`; perform no second call.
2. `NVML_SUCCESS` with returned `count=0`: record a successful empty list;
   perform no second call.
3. `NVML_ERROR_INSUFFICIENT_SIZE` with returned `count>0`: continue to the
   second call below.

Any other return code, a nonzero count in cases 1 or 2, or any second-call
fields in cases 1 or 2 invalidates B2.

### 3.2 Second call

After case 3, checked arithmetic must prove that `count * sizeof(unsigned int)`
is representable by `size_t`. Allocate exactly `count` `unsigned int` elements,
zero every byte, set the second input count to exactly that capacity, and call
the same API with the non-NULL buffer. Record the input capacity,
`second_pointer_null=false`, exact second return code, returned count, and the
complete allocated buffer in its original index order. Allocation failure
invalidates B2.

Require `NVML_SUCCESS` and `1 <= returned_count <= input_capacity`. A second
`NVML_ERROR_INSUFFICIENT_SIZE`, `NVML_ERROR_NOT_SUPPORTED`, or any other return
invalidates B2; it never restarts the pair. Require every unused buffer element
at index `returned_count` or later to remain zero. Publish only the first
`returned_count` values. Every published value must be an exact unsigned
32-bit integer, and duplicate clock values invalidate the list.

The no-retry rule is deliberately fail-closed. A list that changes between the
size probe and the value call is not relabeled stable by repeated observation.

## 4. Memory-to-graphics composition and order

First apply Section 3 to `nvmlDeviceGetSupportedMemoryClocks`. If it returns
`not_supported` or a successful empty list, perform no graphics-clock calls.

Otherwise, canonicalize the distinct returned memory-clock values by the
bytewise ascending order of their ASCII canonical decimal strings. In exactly
that order, apply Section 3 once to
`nvmlDeviceGetSupportedGraphicsClocks(device,memoryClockMHz,...)` for every
memory-clock value and for no other value. Each graphics transcript records its
memory-clock key. A corresponding graphics query may itself record first-call
`NVML_ERROR_NOT_SUPPORTED` under Section 3.1; any other failed transcript is
invalid.

Within every successful memory or graphics list, canonicalize the distinct
clock values by the same bytewise ascending ASCII-decimal order. The final
record contains exactly one graphics transcript per successful memory-clock
value, sorted by that bytewise key. Missing, duplicate, or extra graphics keys
invalidate B2.

## 5. Non-claims and authorization

This amendment resolves only a pre-run ABI and canonicalization ambiguity. It
does not establish that NVML loads, that either API is supported, that a stable
clock list exists, that the target GPU is present, or that B2 is physically
valid or favorable. It authorizes no GPU sampling, GPU execution, instance
creation, or measurement. If independently approved, it authorizes only CPU
implementation and falsification of this exact call transcript and ordering.
