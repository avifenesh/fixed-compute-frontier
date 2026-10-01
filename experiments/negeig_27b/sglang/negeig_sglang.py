"""Widened-beta gate (negeig) for Qwen3.8-27B under SGLang 0.5.20, as a runtime plugin.

Per Gated DeltaNet layer the trained gate replaces the stock beta = sigmoid(b) with

    s = sigmoid(b)            (bf16, exactly as the stock kernels round it)
    t = tanh(W x)             (fp32, one row of W per value head, W zero-initialised)
    beta = s + t * (2 - s if t >= 0 else s)

so beta == s bit for bit at W = 0, beta tends to 2 as t tends to 1 and to 0 as t tends to -1. The delta rule
multiplies both the value and the state readout by beta, so the factor cannot move elsewhere: the two fla kernels
that read beta must receive the final number.

How it is wired (nothing in the SGLang install is edited on disk):

* The plugin entry point `negeig_sglang:register` (group `sglang.srt.plugins`, env `SGLANG_PLUGINS=negeig`) runs
  in every scheduler process before the model loads.
* `Qwen3_5GatedDeltaNet.__init__` is wrapped: each layer gets a bare fp32 parameter `negeig_w`
  [num_v_heads / attn_tp_size, hidden] with a `weight_loader`, zero filled.
* `Qwen3_5GatedDeltaNet.forward` is rebuilt from its own source text with two inserted lines: the producer
  `b = self._negeig_apply(hidden_states, b)` just before the backend call, and the fused decode projection path
  switched off for gated layers (it would hand the backend an unsplit tuple). Every other line is SGLang's.
* The producer is one opaque torch custom op (`negeig::beta`): fp32 `tanh(W x)` plus a Triton combine kernel. Its
  output, the FINAL beta in the dtype of b, travels through the stock `b` slot.
* The two kernels that turn `b` into beta are swapped, for gated layers only, for source-rewritten copies with
  the sigmoid removed: `fused_gdn_gating` (prefill and mixed extend) and the packed single-token decode kernel.
  Everything that cannot take a final beta (generic decode, target verify, replay-SSM, FlashInfer and CuTe DSL
  decode) raises for a gated layer instead of silently computing a stock sigmoid.
* Gates load through `load_weights`: `NEGEIG_GATES=/path/gates.safetensors` (after the stock load, once per model
  instance), `NEGEIG_GATES=zeros` (the stock model, used for the parity gate and as the RL start), or
  `NEGEIG_GATES=stream` (every gate must arrive in the weight stream). Stream keys
  `...layers.N.linear_attn.negeig_w[.weight]` are also accepted in every mode, so RL weight sync reaches the gate.

CLI (CPU, no SGLang import unless stated):

    python negeig_sglang.py static-check [--src DIR]       rewrite targets exist exactly once, rewritten forward compiles
    python negeig_sglang.py export-gates --trainable trainable_000600.pt --out gates.safetensors [--config config.json]
    python negeig_sglang.py install-plugin --dir DIR        copy this file and a dist-info so SGLANG_PLUGINS=negeig works
    python negeig_sglang.py merge-lora --base BASE --trainable trainable_000600.pt --out OUT   (LoRA into the base)

Pinned: SGLang 0.5.20 (the install is refused on any other version unless NEGEIG_ALLOW_SGLANG_VERSION=1).
GPU behaviour (custom op under CUDA graphs, bit identity at W = 0, tensor parallel) is verified by check_parity.py
on the box, not here.
"""

import argparse
import ast
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import textwrap
import threading
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # the CLI and the pure functions work without triton
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAVE_TRITON = False

logger = logging.getLogger("negeig")

PATCH_VERSION = "1"
PINNED_SGLANG = "0.5.20"
ENV_GATES = "NEGEIG_GATES"
ENV_ALLOW_VERSION = "NEGEIG_ALLOW_SGLANG_VERSION"
ENV_ALLOW_FLASHINFER_PREFILL = "NEGEIG_ALLOW_FLASHINFER_PREFILL"
ENV_ALLOW_LOWP_W = "NEGEIG_ALLOW_LOWP_W"
ENV_ALLOW_NON_CUDA = "NEGEIG_ALLOW_NON_CUDA"
ENV_CACHE_DIR = "NEGEIG_CACHE_DIR"

ROW_CHUNK = 8192  # rows of hidden states per fp32 matmul (bounds the fp32 copy of x)

# Every rewrite target must occur exactly once in the pinned source.
FWD_FLAG_LINE = "_gdn_decode_fused_proj_conv"
FWD_CALL_LINE = "attn_result = self.attn("
GATING_OLD = "blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))"
GATING_NEW = "blk_beta_output = blk_b.to(tl.float32)"
PACKED_OLD = "beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
PACKED_NEW = "beta_val = b_val"

MOD_QWEN = "sglang.srt.models.qwen3_5"
MOD_GB = "sglang.srt.layers.attention.linear.gdn_backend"
MOD_GT = "sglang.srt.layers.attention.linear.kernels.gdn_triton"
MOD_GATING = "sglang.kernels.ops.attention.fla.fused_gdn_gating"
MOD_RECURRENT = "sglang.kernels.ops.attention.fla.fused_recurrent"
REL_QWEN = "srt/models/qwen3_5.py"
REL_GATING = "kernels/ops/attention/fla/fused_gdn_gating.py"
REL_RECURRENT = "kernels/ops/attention/fla/fused_recurrent.py"

GATE_KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.linear_attn\.negeig_w(?:\.weight)?$")
GATE_TAP_SUFFIX = ".linear_attn.negeig_w.weight"


class NegeigPatchError(SystemExit):
    """Registration or install failed. A SystemExit so SGLang's plugin loader (which swallows Exception) cannot hide it."""

    def __init__(self, msg: str):
        super().__init__("negeig: " + msg)
        self.msg = msg


class NegeigError(RuntimeError):
    """A load or runtime guard fired."""


def _say(msg: str) -> None:
    line = f"negeig: {msg}"
    print(line, file=sys.stderr, flush=True)
    logger.info(line)


# ----------------------------------------------------------------------------------------------------------
# the arithmetic (pure functions, CPU testable)
# ----------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _highest_matmul() -> Iterator[None]:
    """fp32 matmuls at full precision around the gate projection (SGLang can enable TF32 globally)."""
    prev = torch.get_float32_matmul_precision()
    if prev == "highest":
        yield
        return
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def gate_t(x: torch.Tensor, w: torch.Tensor, chunk: int = ROW_CHUNK) -> torch.Tensor:
    """t = tanh(W x) in fp32, [T, nv]. x is the normalised hidden state fed to the layer, w is fp32 [nv, hidden]."""
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    with _highest_matmul():
        if x.shape[0] <= chunk:
            return torch.tanh(F.linear(x.float(), w))
        out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
        for i in range(0, x.shape[0], chunk):
            out[i : i + chunk] = torch.tanh(F.linear(x[i : i + chunk].float(), w))
        return out


def beta_reference(b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The widened beta in torch, same arithmetic as the HF patch: s rounded to b's dtype, combine in fp32."""
    s = torch.sigmoid(b.float()).to(b.dtype).float()
    coef = torch.where(t >= 0, 2.0 - s, s)
    return (s + t.float() * coef).to(b.dtype)


if HAVE_TRITON:

    @triton.jit
    def _negeig_beta_kernel(
        b_ptr,
        t_ptr,
        o_ptr,
        stride_b,
        stride_t,
        stride_o,
        NV: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK)
        m = col < NV
        b = tl.load(b_ptr + row * stride_b + col, mask=m, other=0.0).to(tl.float32)
        t = tl.load(t_ptr + row * stride_t + col, mask=m, other=0.0)
        # the stock kernels round the sigmoid to b's dtype before it multiplies anything
        s = tl.sigmoid(b).to(o_ptr.dtype.element_ty).to(tl.float32)
        coef = tl.where(t >= 0.0, 2.0 - s, s)
        beta = s + t * coef
        tl.store(o_ptr + row * stride_o + col, beta.to(o_ptr.dtype.element_ty), mask=m)


def _triton_ok(b: torch.Tensor) -> bool:
    return HAVE_TRITON and (b.is_cuda or os.environ.get("TRITON_INTERPRET") == "1")


def beta_combine(b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The final beta [T, nv] in b's dtype, always a NEW contiguous tensor (b may be a view of a shared buffer)."""
    if b.dim() != 2 or b.shape != t.shape:
        raise NegeigError(f"beta_combine shape mismatch: b {tuple(b.shape)} vs t {tuple(t.shape)}")
    if not _triton_ok(b):
        return beta_reference(b, t).contiguous()
    out = torch.empty(b.shape, dtype=b.dtype, device=b.device)
    rows, nv = b.shape
    if rows == 0:
        return out
    if b.stride(-1) != 1:
        b = b.contiguous()
    t = t.contiguous()
    _negeig_beta_kernel[(rows,)](
        b, t, out, b.stride(0), t.stride(0), out.stride(0), NV=nv, BLOCK=triton.next_power_of_2(nv)
    )
    return out


def beta_eager(b: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != b.shape[0]:
        raise NegeigError(f"row mismatch between hidden states {tuple(x.shape)} and b {tuple(b.shape)}")
    if w.dtype != torch.float32:
        raise NegeigError(f"negeig_w must stay fp32, got {w.dtype}")
    return beta_combine(b, gate_t(x, w))


_OP_LOCK = threading.Lock()
_OP_STATE: Dict[str, object] = {}


def beta_op():
    """`torch.ops.negeig.beta(b, x, w)`, defined once per process. Opaque to torch.compile, so piecewise
    graphs and CUDA graph capture both see one custom op whose fake output is an empty tensor shaped like b."""
    with _OP_LOCK:
        if "fn" in _OP_STATE:
            return _OP_STATE["fn"]
        fn = None
        if hasattr(torch.library, "custom_op"):
            try:
                fn = torch.ops.negeig.beta  # already defined by an earlier import of this module
            except (AttributeError, RuntimeError):
                fn = None
            if fn is None:

                @torch.library.custom_op("negeig::beta", mutates_args=())
                def _beta(b: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
                    return beta_eager(b, x, w)

                @_beta.register_fake
                def _beta_fake(b, x, w):
                    return torch.empty(b.shape, dtype=b.dtype, device=b.device)

                fn = torch.ops.negeig.beta
        _OP_STATE["fn"] = fn if fn is not None else beta_eager
        return _OP_STATE["fn"]


# ----------------------------------------------------------------------------------------------------------
# source rewriting (pure string work, checked against the pinned source by static-check and again at install)
# ----------------------------------------------------------------------------------------------------------
def rewrite_forward_source(src: str) -> str:
    """The stock `Qwen3_5GatedDeltaNet.forward` text with the producer inserted and the fused decode path gated."""
    src = textwrap.dedent(src).rstrip("\n")
    lines = src.split("\n")
    if not lines[0].startswith("def forward("):
        raise NegeigPatchError(f"forward source does not start with 'def forward(': {lines[0]!r}")
    out: List[str] = []
    n_flag = n_call = 0
    for ln in lines:
        s = ln.strip()
        indent = ln[: len(ln) - len(ln.lstrip())]
        if s == FWD_FLAG_LINE:
            out.append(indent + FWD_FLAG_LINE + ' and not getattr(self, "_negeig", False)')
            n_flag += 1
            continue
        if s == FWD_CALL_LINE:
            out.append(indent + 'if getattr(self, "_negeig", False):')
            out.append(indent + "    b = self._negeig_apply(hidden_states, b)")
            n_call += 1
        out.append(ln)
    if n_flag != 1 or n_call != 1:
        raise NegeigPatchError(
            f"forward rewrite targets changed upstream: {FWD_FLAG_LINE!r} x{n_flag}, {FWD_CALL_LINE!r} x{n_call}"
        )
    return "from __future__ import annotations\n" + "\n".join(out) + "\n"


def rewrite_kernel_source(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    if n != 1:
        raise NegeigPatchError(f"{what}: rewrite target occurs {n} times, expected 1: {old!r}")
    return src.replace(old, new)


def _cache_dir() -> Path:
    base = os.environ.get(ENV_CACHE_DIR)
    p = Path(base) if base else Path(tempfile.gettempdir()) / f"negeig_sglang_{os.getuid()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_cached(name: str, text: str) -> Path:
    """Triton needs a real source file, so every generated module lives in a content-hash-named cache file."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = _cache_dir() / f"{name}_{h}.py"
    if not path.exists():
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, path)
    return path


def _load_file_module(modname: str, path: Path):
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod  # before exec: triton resolves the defining module through sys.modules
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    return mod


def patched_kernel_module(orig_module, old: str, new: str, tag: str):
    """A source-rewritten copy of an installed fla kernel file, imported under its own module name."""
    src = Path(orig_module.__file__).read_text()
    text = f"# negeig passthrough copy of {orig_module.__name__} (generated, do not edit)\n" + rewrite_kernel_source(
        src, old, new, orig_module.__name__
    )
    path = _write_cached(f"negeig_{tag}", text)
    return _load_file_module(f"negeig_patched_{tag}_{path.stem.rsplit('_', 1)[-1]}", path)


# ----------------------------------------------------------------------------------------------------------
# install
# ----------------------------------------------------------------------------------------------------------
_INSTALL_LOCK = threading.Lock()
_INSTALLED: Dict[str, object] = {}
_TLS = threading.local()


def _marked(obj) -> bool:
    return bool(getattr(getattr(obj, "A_log", obj), "_negeig", False))


def _check_version() -> None:
    try:
        import sglang  # noqa: F401

        ver = getattr(sys.modules["sglang"], "__version__", None)
    except Exception as e:
        raise NegeigPatchError(f"cannot import sglang: {e}")
    if ver is None:
        try:
            from importlib.metadata import version

            ver = version("sglang")
        except Exception:
            ver = "unknown"
    if ver != PINNED_SGLANG and os.environ.get(ENV_ALLOW_VERSION) != "1":
        raise NegeigPatchError(
            f"this patch is pinned to sglang {PINNED_SGLANG}, found {ver} "
            f"(set {ENV_ALLOW_VERSION}=1 only after re-running static-check and check_parity on it)"
        )


def _import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise NegeigPatchError(f"cannot import {name}: {e!r}")


def _make_gating_wrapper(orig, patched):
    @functools.wraps(orig)
    def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
        if getattr(A_log, "_negeig", False):
            return patched(A_log, a, b, dt_bias, beta, threshold)
        return orig(A_log, a, b, dt_bias, beta, threshold)

    fused_gdn_gating._negeig_wrapped = True
    return fused_gdn_gating


def _make_packed_wrapper(orig, patched):
    @functools.wraps(orig)
    def fused_recurrent_gated_delta_rule_packed_decode(*args, **kwargs):
        A_log = kwargs.get("A_log", args[3] if len(args) > 3 else None)
        if getattr(A_log, "_negeig", False):
            return patched(*args, **kwargs)
        return orig(*args, **kwargs)

    fused_recurrent_gated_delta_rule_packed_decode._negeig_wrapped = True
    return fused_recurrent_gated_delta_rule_packed_decode


def _make_refusing_wrapper(orig, what: str):
    @functools.wraps(orig)
    def refusing(*args, **kwargs):
        A_log = kwargs.get("A_log", args[0] if args else None)
        if getattr(A_log, "_negeig", False):
            raise NegeigError(
                f"{what} computes sigmoid(b) inside the kernel and cannot take the widened beta; "
                "run with --linear-attn-backend triton, no --enable-linear-replayssm and no speculative decoding"
            )
        return orig(*args, **kwargs)

    refusing._negeig_wrapped = True
    return refusing


def _install_kernel_swaps(GB, GT) -> None:
    gating_mod = _import(MOD_GATING)
    rec_mod = _import(MOD_RECURRENT)
    if getattr(GB, "fused_gdn_gating", None) is None or getattr(GB.fused_gdn_gating, "__module__", "") != MOD_GATING:
        raise NegeigPatchError("gdn_backend.fused_gdn_gating is not the stock fla kernel wrapper (CPU, NPU or XPU build?)")
    p_gating = patched_kernel_module(gating_mod, GATING_OLD, GATING_NEW, "fused_gdn_gating")
    p_rec = patched_kernel_module(rec_mod, PACKED_OLD, PACKED_NEW, "fused_recurrent")
    GB.fused_gdn_gating = _make_gating_wrapper(gating_mod.fused_gdn_gating, p_gating.fused_gdn_gating)
    for name in (
        "fused_recurrent_gated_delta_rule_packed_decode",
        "fused_recurrent_gdn_replayssm_decode",
        "fused_sigmoid_gating_delta_rule_update",
    ):
        if not hasattr(GT, name):
            raise NegeigPatchError(f"gdn_triton has no {name}: the Triton GDN kernel set is not importable here")
    GT.fused_recurrent_gated_delta_rule_packed_decode = _make_packed_wrapper(
        GT.fused_recurrent_gated_delta_rule_packed_decode, p_rec.fused_recurrent_gated_delta_rule_packed_decode
    )
    GT.fused_recurrent_gdn_replayssm_decode = _make_refusing_wrapper(
        GT.fused_recurrent_gdn_replayssm_decode, "the replay-SSM decode kernel"
    )
    GT.fused_sigmoid_gating_delta_rule_update = _make_refusing_wrapper(
        GT.fused_sigmoid_gating_delta_rule_update, "the generic fused sigmoid-gating decode kernel"
    )
    # the same kernel name imported lazily by the target-verify helpers resolves in its own module
    sig_mod = importlib.import_module("sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent")
    if not getattr(sig_mod.fused_sigmoid_gating_delta_rule_update, "_negeig_wrapped", False):
        sig_mod.fused_sigmoid_gating_delta_rule_update = _make_refusing_wrapper(
            sig_mod.fused_sigmoid_gating_delta_rule_update, "the fused sigmoid-gating kernel"
        )


def _install_backend_guards(GB, GT) -> None:
    cls = GB.GDNAttnBackend
    orig_decode, orig_extend = cls.forward_decode, cls.forward_extend
    Triton = GT.TritonGDNKernel

    def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        if _marked(layer):
            if isinstance(mixed_qkv, tuple):
                raise NegeigError("fused GDN decode projection reached a gated layer (mixed_qkv is a tuple)")
            if not isinstance(self.kernel_dispatcher.decode_kernel, Triton):
                raise NegeigError(
                    "gated layers need --linear-attn-decode-backend triton "
                    f"(decode kernel is {type(self.kernel_dispatcher.decode_kernel).__name__})"
                )
        return orig_decode(self, layer=layer, forward_batch=forward_batch, mixed_qkv=mixed_qkv, a=a, b=b, **kwargs)

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        if _marked(layer):
            if forward_batch.forward_mode.is_target_verify():
                raise NegeigError("speculative target verify is not supported with the widened beta gate")
            if (
                not isinstance(self.kernel_dispatcher.extend_kernel, Triton)
                and os.environ.get(ENV_ALLOW_FLASHINFER_PREFILL) != "1"
            ):
                raise NegeigError(
                    "gated layers need --linear-attn-prefill-backend triton: the FlashInfer and CuTe DSL prefill kernels "
                    f"are not verified for beta above 1 (set {ENV_ALLOW_FLASHINFER_PREFILL}=1 to try them)"
                )
        return orig_extend(self, layer=layer, forward_batch=forward_batch, mixed_qkv=mixed_qkv, a=a, b=b, **kwargs)

    forward_decode._negeig_wrapped = True
    forward_extend._negeig_wrapped = True
    cls.forward_decode = forward_decode
    cls.forward_extend = forward_extend


def _shard_gate(mod, w: torch.Tensor) -> torch.Tensor:
    """This rank's rows of a full [num_v_heads, hidden] gate, as fp32."""
    want = (mod.num_v_heads, mod.hidden_size)
    if tuple(w.shape) != want:
        raise NegeigError(f"layer {mod.layer_id}: gate shape {tuple(w.shape)}, expected the full {want}")
    if w.dtype != torch.float32 and os.environ.get(ENV_ALLOW_LOWP_W) != "1":
        raise NegeigError(
            f"layer {mod.layer_id}: gate arrived as {w.dtype}; it must stay fp32 end to end "
            f"(set {ENV_ALLOW_LOWP_W}=1 to accept and upcast)"
        )
    nv_tp = mod.num_v_heads // mod.attn_tp_size
    r = mod.attn_tp_rank
    return w[r * nv_tp : (r + 1) * nv_tp].to(torch.float32)


def _make_weight_loader(mod):
    def weight_loader(param, loaded_weight, *args, **kwargs):
        t = _shard_gate(mod, loaded_weight)
        with torch.no_grad():
            param.data.copy_(t.to(param.device))
        param._negeig_loaded = True

    return weight_loader


def _install_gdn_class(M) -> None:
    cls = M.Qwen3_5GatedDeltaNet
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        tp = self.attn_tp_size
        if self.num_v_heads % tp:
            raise NegeigPatchError(f"num_v_heads {self.num_v_heads} is not divisible by attn_tp_size {tp}")
        if self.attn.A_log is not self.A_log:
            raise NegeigPatchError("RadixLinearAttention.A_log is not the layer's A_log Parameter; the marker would not reach the kernels")
        w = nn.Parameter(
            torch.zeros(self.num_v_heads // tp, self.hidden_size, dtype=torch.float32), requires_grad=False
        )
        w.weight_loader = _make_weight_loader(self)
        self.negeig_w = w
        self.A_log._negeig = True  # read by the kernel wrappers and the backend guards
        self._negeig = True  # read by the rewritten forward

    op = beta_op()  # resolved once here: beta_op() takes a lock, which torch.compile(fullgraph=True) cannot trace

    def _negeig_apply(self, x, b):
        if isinstance(x, (tuple, list)):
            raise NegeigError("the gated layer received a pre-quantised hidden state tuple")
        if self.negeig_w.dtype != torch.float32:
            raise NegeigError(f"negeig_w must stay fp32, got {self.negeig_w.dtype}")
        return op(b, x, self.negeig_w)

    src = inspect_source(cls.forward)
    new_src = rewrite_forward_source(src)
    path = _write_cached("negeig_forward", new_src)
    ns: dict = {}
    exec(compile(new_src, str(path), "exec"), M.__dict__, ns)
    new_forward = ns["forward"]
    new_forward._negeig_wrapped = True
    __init__._negeig_wrapped = True
    cls.__init__ = __init__
    cls._negeig_apply = _negeig_apply
    cls.forward = new_forward


def inspect_source(fn) -> str:
    import inspect

    return inspect.getsource(fn)


def _gate_mode() -> str:
    v = os.environ.get(ENV_GATES, "").strip()
    if not v:
        raise NegeigError(f"{ENV_GATES} is not set: a safetensors path, 'zeros' or 'stream'")
    return v


def gated_modules(model, M) -> List:
    return [m for m in model.modules() if isinstance(m, M.Qwen3_5GatedDeltaNet) and getattr(m, "_negeig", False)]


def read_gates_file(path: str) -> Dict[int, torch.Tensor]:
    from safetensors import safe_open

    out: Dict[int, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not a Mapping)
            m = GATE_KEY_RE.search(k)
            if not m:
                raise NegeigError(f"{path}: unexpected key {k!r} (gate keys look like ...layers.N.linear_attn.negeig_w)")
            lid = int(m.group(1))
            if lid in out:
                raise NegeigError(f"{path}: layer {lid} appears twice")
            out[lid] = f.get_tensor(k)
    if not out:
        raise NegeigError(f"{path}: no gate tensors")
    return out


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _after_load(model, M) -> None:
    """Runs once per model instance after the first stock load: gates from the file, zeros, or the stream."""
    if getattr(model, "_negeig_done", False):
        return
    mods = gated_modules(model, M)
    if not mods:
        raise NegeigError("the model has no gated GDN layers: the patch did not reach this model class")
    mode = _gate_mode()
    with torch.no_grad():
        if mode == "zeros":
            for m in mods:
                if not getattr(m.negeig_w, "_negeig_loaded", False):
                    m.negeig_w.zero_()  # the capture-safe loader leaves a finite sentinel here; zero it on purpose
            how = "zeros (layers absent from the stream)"
        elif mode == "stream":
            missing = sorted(m.layer_id for m in mods if not getattr(m.negeig_w, "_negeig_loaded", False))
            if missing:
                raise NegeigError(f"{ENV_GATES}=stream but no gate arrived in the weight stream for layers {missing}")
            how = "stream"
        else:
            if not os.path.isfile(mode):
                raise NegeigError(f"{ENV_GATES}={mode!r} is neither 'zeros', 'stream' nor a file")
            gates = read_gates_file(mode)
            missing = sorted(m.layer_id for m in mods if m.layer_id not in gates)
            if missing:
                raise NegeigError(f"{mode} has no gate for GDN layers {missing}")
            for m in mods:
                t = _shard_gate(m, gates[m.layer_id])
                m.negeig_w.data.copy_(t.to(m.negeig_w.device))
                m.negeig_w._negeig_loaded = True
            how = f"file {mode} sha256={file_sha256(mode)}"
        absmax = max(float(m.negeig_w.detach().abs().max()) for m in mods)
    model._negeig_done = True
    _say(f"active v{PATCH_VERSION}: {len(mods)} gated GDN layers on this rank, gates from {how}, max|W|={absmax:.6g}")


def _tap(weights: Iterable[Tuple[str, torch.Tensor]]):
    """Accept the trainer's key form `...linear_attn.negeig_w.weight` next to the param's own `...negeig_w`."""
    for name, w in weights:
        if name.endswith(GATE_TAP_SUFFIX):
            name = name[: -len(".weight")]
        yield name, w


def _install_load_weights(M) -> None:
    for cname in (
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        cls = getattr(M, cname, None)
        if cls is None or "load_weights" not in cls.__dict__:
            raise NegeigPatchError(f"{cname} does not define its own load_weights in this SGLang build")
        orig = cls.__dict__["load_weights"]

        def make(orig):
            @functools.wraps(orig)
            def load_weights(self, weights):
                if getattr(_TLS, "depth", 0) > 0:
                    return orig(self, weights)
                _TLS.depth = 1
                try:
                    loaded = orig(self, _tap(weights))
                    _after_load(self, M)
                    for m in gated_modules(self, M):
                        if loaded is not None:
                            loaded.add(f"model.layers.{m.layer_id}.linear_attn.negeig_w")
                finally:
                    _TLS.depth = 0
                return loaded

            load_weights._negeig_wrapped = True
            return load_weights

        cls.load_weights = make(orig)


def install() -> None:
    """Idempotent. Raises NegeigPatchError on anything that is not exactly the pinned SGLang."""
    with _INSTALL_LOCK:
        if _INSTALLED.get("done"):
            return
        if not torch.cuda.is_available() and os.environ.get(ENV_ALLOW_NON_CUDA) != "1":
            raise NegeigPatchError(f"no CUDA device (set {ENV_ALLOW_NON_CUDA}=1 for CPU wiring tests only)")
        _check_version()
        M = _import(MOD_QWEN)
        GB = _import(MOD_GB)
        GT = _import(MOD_GT)
        beta_op()
        try:
            _install_kernel_swaps(GB, GT)
            _install_backend_guards(GB, GT)
            _install_gdn_class(M)
            _install_load_weights(M)
        except (NegeigPatchError, NegeigError):
            raise
        except Exception as e:
            raise NegeigPatchError(f"install failed: {type(e).__name__}: {e}")
        _INSTALLED["done"] = True
        _say(f"installed v{PATCH_VERSION} on sglang {PINNED_SGLANG} (pid {os.getpid()})")


def register() -> None:
    """The `sglang.srt.plugins` entry point."""
    _gate_mode_or_die()
    install()


def _gate_mode_or_die() -> None:
    try:
        _gate_mode()
    except NegeigError as e:
        raise NegeigPatchError(str(e))


# ----------------------------------------------------------------------------------------------------------
# CLI: static-check
# ----------------------------------------------------------------------------------------------------------
def _find_src_root(src: Optional[str]) -> Path:
    if src:
        root = Path(src)
    else:
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("static-check: sglang is not importable, pass --src DIR (an unpacked sglang package)")
        root = Path(list(spec.submodule_search_locations)[0])
    if not (root / REL_QWEN).is_file():
        raise SystemExit(f"static-check: {root / REL_QWEN} not found")
    return root


def _forward_segment(qwen_text: str) -> str:
    tree = ast.parse(qwen_text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5GatedDeltaNet":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
                    if fn.decorator_list:
                        raise NegeigPatchError("Qwen3_5GatedDeltaNet.forward is decorated upstream: rewrite it by hand")
                    seg = ast.get_source_segment(qwen_text, fn)
                    return textwrap.dedent(" " * fn.col_offset + seg)
    raise NegeigPatchError("Qwen3_5GatedDeltaNet.forward not found")


def static_check(src: Optional[str] = None) -> dict:
    root = _find_src_root(src)
    qwen = (root / REL_QWEN).read_text()
    new_fwd = rewrite_forward_source(_forward_segment(qwen))
    compile(new_fwd, "<negeig_forward>", "exec")
    gating = rewrite_kernel_source((root / REL_GATING).read_text(), GATING_OLD, GATING_NEW, REL_GATING)
    packed = rewrite_kernel_source((root / REL_RECURRENT).read_text(), PACKED_OLD, PACKED_NEW, REL_RECURRENT)
    compile(gating, REL_GATING, "exec")
    compile(packed, REL_RECURRENT, "exec")
    for cname in ("Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"):
        m = re.search(rf"^class {cname}\b.*?(?=^class |\Z)", qwen, re.S | re.M)
        if not m or not re.search(r"^    def load_weights\(", m.group(0), re.M):
            raise NegeigPatchError(f"{cname}: no own load_weights")
    ver = None
    vfile = root / "_version.py"
    if vfile.is_file():
        mm = re.search(r"version\s*=\s*['\"]([^'\"]+)", vfile.read_text())
        ver = mm.group(1) if mm else None
    return {
        "src": str(root),
        "sglang_version": ver,
        "pinned": PINNED_SGLANG,
        "forward_rewrite_lines_added": new_fwd.count("\n") - qwen_forward_lines(qwen) - 1,
        "gating_target_count": 1,
        "packed_decode_target_count": 1,
        "load_weights_classes": 4,
        "ok": ver in (None, PINNED_SGLANG),
    }


def qwen_forward_lines(qwen_text: str) -> int:
    return _forward_segment(qwen_text).rstrip("\n").count("\n") + 1


# ----------------------------------------------------------------------------------------------------------
# CLI: export-gates
# ----------------------------------------------------------------------------------------------------------
def _flatten_trainable(obj) -> dict:
    if isinstance(obj, dict) and "trainable" in obj and isinstance(obj["trainable"], dict):
        return obj["trainable"]
    if isinstance(obj, dict):
        return obj
    raise SystemExit("export-gates: the file is neither a trainable dict nor a ckpt with a 'trainable' key")


def export_gates(trainable: str, out: str, config: Optional[str] = None) -> dict:
    from safetensors.torch import save_file

    state = _flatten_trainable(torch.load(trainable, map_location="cpu", weights_only=True))
    if any(re.search(r"negeig_wa(\.|$)", k) for k in state):
        raise SystemExit("export-gates: the file carries decay-gate params (negeig_wa); this patch serves the beta gate only")
    tensors: Dict[str, torch.Tensor] = {}
    ids = []
    for k, v in state.items():
        m = GATE_KEY_RE.search(k)
        if not m:
            continue
        lid = int(m.group(1))
        if lid in ids:
            raise SystemExit(f"export-gates: layer {lid} appears twice")
        if v.dim() != 2:
            raise SystemExit(f"export-gates: {k} has shape {tuple(v.shape)}, expected [num_v_heads, hidden]")
        if not torch.isfinite(v.float()).all():
            raise SystemExit(f"export-gates: {k} has non-finite values")
        ids.append(lid)
        tensors[f"model.language_model.layers.{lid}.linear_attn.negeig_w"] = v.detach().to(torch.float32).contiguous()
    if not tensors:
        raise SystemExit("export-gates: no negeig_w tensors in the file")
    shapes = {tuple(v.shape) for v in tensors.values()}
    if len(shapes) != 1:
        raise SystemExit(f"export-gates: gate shapes differ: {sorted(shapes)}")
    if config:
        cfg = json.loads(Path(config).read_text())
        cfg = cfg.get("text_config", cfg)
        want = sorted(i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention")
        if sorted(ids) != want:
            raise SystemExit(f"export-gates: layers {sorted(set(want) ^ set(ids))} differ from the config's linear_attention layers")
    meta = {
        "format": "negeig-gates-v1",
        "source": os.path.basename(trainable),
        "source_sha256": file_sha256(trainable),
        "n_layers": str(len(ids)),
        "layers": ",".join(str(i) for i in sorted(ids)),
        "shape": "x".join(str(d) for d in next(iter(shapes))),
        "dtype": "float32",
    }
    save_file(tensors, out, metadata=meta)
    return {"out": out, "sha256": file_sha256(out), **meta}


# ----------------------------------------------------------------------------------------------------------
# CLI: install-plugin
# ----------------------------------------------------------------------------------------------------------
def install_plugin(dest: str) -> dict:
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), d / "negeig_sglang.py")
    di = d / f"negeig_sglang-{PATCH_VERSION}.0.dist-info"
    di.mkdir(exist_ok=True)
    (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: negeig-sglang\nVersion: {PATCH_VERSION}.0\n")
    (di / "entry_points.txt").write_text("[sglang.srt.plugins]\nnegeig = negeig_sglang:register\n")
    return {"dir": str(d), "launch_env": {"PYTHONPATH": str(d), "SGLANG_PLUGINS": "negeig", ENV_GATES: "<gates.safetensors|zeros|stream>"}}


# ----------------------------------------------------------------------------------------------------------
# CLI: merge-lora
# ----------------------------------------------------------------------------------------------------------
LORA_KEY = re.compile(r"^(?:.*?\.)?(?P<module>layers\.\d+\..+?)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$")


def stochastic_bf16(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """fp32 to bf16, rounding up in magnitude with probability equal to the dropped fraction (unbiased)."""
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, generator=generator)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


def merge_lora(base: str, trainable: str, out: str, scale: float = 2.0, rounding: str = "stochastic",
               seed: int = 20261001, min_delta_kept: float = 0.95) -> dict:
    """Fold W += scale * B @ A into the base bf16 safetensors. Reports delta_kept, the projection of what the bf16
    weights moved onto the intended delta (1 keeps all of it), and refuses to write a merge below min_delta_kept."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    basep, outp = Path(base), Path(out)
    state = _flatten_trainable(torch.load(trainable, map_location="cpu", weights_only=True))
    pairs: Dict[str, dict] = {}
    for k, v in state.items():
        m = LORA_KEY.match(k)
        if m:
            pairs.setdefault(m["module"], {})[m["ab"]] = v.float()
    if not pairs:
        raise SystemExit("merge-lora: no LoRA tensors in the trainable file")
    bad = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if bad:
        raise SystemExit(f"merge-lora: unpaired LoRA tensors for {bad[:3]}")
    idx_path = basep / "model.safetensors.index.json"
    if idx_path.is_file():
        index = json.loads(idx_path.read_text())["weight_map"]
    else:
        with safe_open(basep / "model.safetensors", framework="pt") as f:
            index = {k: "model.safetensors" for k in f.keys()}  # noqa: SIM118
    targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for module, ab in pairs.items():
        tail = "." + module + ".weight"
        hits = [n for n in index if n.endswith(tail) and "visual" not in n and "mtp" not in n]
        if len(hits) != 1:
            raise SystemExit(f"merge-lora: {module} matches base tensors {hits}")
        a, b = ab["A"], ab["B"]
        if a.shape[0] != b.shape[1]:
            raise SystemExit(f"merge-lora: {module} rank mismatch A {tuple(a.shape)} B {tuple(b.shape)}")
        targets[hits[0]] = (a, b)
    outp.mkdir(parents=True, exist_ok=False)
    gen = torch.Generator().manual_seed(seed)
    by_file: Dict[str, List[str]] = {}
    for name, fname in index.items():
        by_file.setdefault(fname, []).append(name)
    kept_num = kept_den = 0.0
    per: Dict[str, float] = {}
    worst = 1.0
    for fname in sorted(by_file):
        names = [n for n in by_file[fname] if n in targets]
        if not names:
            shutil.copy2(basep / fname, outp / fname)
            continue
        tensors = {}
        with safe_open(basep / fname, framework="pt") as f:
            meta = f.metadata()
            for n in f.keys():  # noqa: SIM118
                tensors[n] = f.get_tensor(n)
        for n in names:
            a, b = targets[n]
            w = tensors[n]
            if w.dtype != torch.bfloat16:
                raise SystemExit(f"merge-lora: {n} is {w.dtype}, expected bf16")
            delta = scale * (b @ a)
            if delta.shape != w.shape:
                raise SystemExit(f"merge-lora: {n} delta {tuple(delta.shape)} vs weight {tuple(w.shape)}")
            merged = w.float() + delta
            new = stochastic_bf16(merged, gen) if rounding == "stochastic" else merged.to(torch.bfloat16)
            num = float(((new.float() - w.float()) * delta).sum())
            den = float(delta.pow(2).sum())
            kept_num += num
            kept_den += den
            per[n] = num / den if den > 0 else 1.0
            worst = min(worst, per[n])
            tensors[n] = new
        save_file(tensors, outp / fname, metadata=meta or {"format": "pt"})
    for p in basep.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, outp / p.name)
    kept = kept_num / kept_den if kept_den else 1.0
    receipt = {
        "base": str(basep), "trainable": trainable, "trainable_sha256": file_sha256(trainable),
        "targets": len(targets), "scale": scale, "rounding": rounding,
        "seed": seed if rounding == "stochastic" else None,
        "delta_kept": kept, "delta_kept_worst_tensor": worst, "min_delta_kept": min_delta_kept,
    }
    (outp / "MERGE-RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if kept < min_delta_kept or worst < min_delta_kept * 0.5:
        shutil.rmtree(outp)
        raise SystemExit(
            f"merge-lora: delta_kept {kept:.5f} (worst tensor {worst:.5f}) is below {min_delta_kept}; "
            "the bf16 merge drops the adapter. Use --rounding stochastic, or serve the adapter unmerged elsewhere."
        )
    return receipt


# ----------------------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("static-check")
    p.add_argument("--src", help="unpacked sglang package dir (default: the installed one)")
    p = sub.add_parser("export-gates")
    p.add_argument("--trainable", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="model config.json: verify the gate layers equal its linear_attention layers")
    p = sub.add_parser("install-plugin")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("merge-lora")
    p.add_argument("--base", required=True)
    p.add_argument("--trainable", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=2.0, help="lora_alpha / r (train27.py uses alpha = 2 r)")
    p.add_argument("--rounding", choices=("stochastic", "nearest"), default="stochastic")
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--min-delta-kept", type=float, default=0.95)
    a = ap.parse_args(argv)
    if a.cmd == "static-check":
        res = static_check(a.src)
    elif a.cmd == "export-gates":
        res = export_gates(a.trainable, a.out, a.config)
    elif a.cmd == "install-plugin":
        res = install_plugin(a.dir)
    else:
        res = merge_lora(a.base, a.trainable, a.out, a.scale, a.rounding, a.seed, a.min_delta_kept)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
