# Radial-trust Cayley B2 physical preregistration — Amendment 1

Date: 2026-08-01  
Status: **PROPOSED NORMATIVE AMENDMENT; DO NOT IMPLEMENT OR RUN**

## 1. Parent and scope

This document is a prospective normative amendment to
`radial-trust-cayley-b2-physical-preregistration.md` at frozen SHA-256
`31ed1d8e6cffcd9b189dc4736e55bb5ef175a8bdf3fbcc514dc8ae13891e256e`.
It changes only the previously undefined `<BDF>` placeholders in Section 7.0,
lines 560 and 565. It does not alter any architecture, numerical operator,
artifact, workload, control, statistical gate, timing rule, byte ledger, or
device-acquisition rule.

This amendment is not effective until it passes an independent paper audit.
The effective preregistration identity will then be the ordered hash pair
`(parent_sha256, amendment_1_sha256)`; the historical parent file and prior
slice receipts remain byte-for-byte unchanged.

## 2. Defect

The parent requires reads under
`/sys/bus/pci/devices/<BDF>` and
`/proc/driver/nvidia/gpus/<BDF>` but never defines `<BDF>`.

The pinned CUDA/NVML header
`/usr/local/cuda/include/nvml.h`, SHA-256
`28b51fbd44df16adf1e58229778414a4d1e7e05fdd4a74526ef0affb75f18416`,
defines the `nvmlPciInfo_t.busId` format as `%08X:%02X:%02X.0` and the
`busIdLegacy` format as `%04X:%02X:%02X.0`. Linux PCI sysfs and NVIDIA's
per-GPU proc directory use the four-domain-digit Linux PCI basename. Raw NVML
strings also use uppercase hexadecimal, whereas Linux path bytes are
case-sensitive. Selecting either raw string as a path can therefore produce a
different record or `ENOENT`.

## 3. Normative replacement

In both affected parent paths, `<BDF>` means `path_bdf` below and nothing else.

At every container-invariant snapshot, call
`nvmlDeviceGetPciInfo_v3` before any BDF-dependent filesystem read. Record its
exact NVML return code. A return other than `NVML_SUCCESS`, including
`NVML_ERROR_NOT_SUPPORTED`, invalidates B2 at that snapshot; perform neither
BDF-dependent filesystem read and do not derive or substitute a path. This is
a mandatory identity query and specializes the parent's general unsupported-
query recording rule.

After `NVML_SUCCESS`, the zero-initialized `nvmlPciInfo_t` supplies unsigned
numeric fields `domain`, `bus`, and `device` and two NUL-terminated byte strings
`busId` and `busIdLegacy`. Before either BDF-dependent filesystem read:

1. require `domain <= 0xffff`, `bus <= 0xff`, and `device <= 0x1f`;
2. form the exact ASCII byte strings
   `expected_bus_id = "%08X:%02X:%02X.0" % (domain,bus,device)` and
   `expected_legacy = "%04X:%02X:%02X.0" % (domain,bus,device)`;
3. require `busId` and `busIdLegacy` each to contain a NUL within its declared
   array, require its first NUL to occur exactly after the expected string, and
   require all bytes before that first NUL to equal its expected string
   byte-for-byte; bytes after the first NUL have no string meaning and are
   neither constrained nor published;
4. the literal `.0` checks the only function value represented by the pinned
   NVML structure; any nonzero, missing, or differently encoded function is
   invalid; and
5. form
   `path_bdf = "%04x:%02x:%02x.0" % (domain,bus,device)` as lowercase ASCII.

The invariant records decimal-string `domain`, `bus`, and `device`; base64url-
without-padding encodings of the validated pre-NUL `busId` and `busIdLegacy`
strings; and base64url-without-padding `path_bdf`.

These fields are additive to the parent-required complete
`nvmlDeviceGetPciInfo_v3` result, including `pciDeviceId` and
`pciSubSystemId`; they do not replace or omit any other frozen NVML field.

The exact filesystem paths are then:

```text
/sys/bus/pci/devices/<path_bdf>
/proc/driver/nvidia/gpus/<path_bdf>/information
```

No directory scan, case-insensitive lookup, existence-selected fallback,
shortened domain, raw `busId`, raw `busIdLegacy`, `lspci`, CUDA ordinal, NVML
index, or provider metadata may choose or repair the path. Failure of either
exact path is encoded under the parent rule as its exact `errno`; inconsistency
among numeric fields, raw strings, or `path_bdf` invalidates B2.

## 4. Non-claims and authorization

This amendment resolves only a byte-level identity ambiguity. It does not
establish that the target GPU is present, that NVML succeeds, that any PCI file
is readable, or that B2 is physically valid or favorable. It authorizes no GPU
sampling, GPU execution, instance creation, or measurement. If independently
approved, it authorizes only CPU implementation and falsification of the
amended path derivation and platform-record schema.
