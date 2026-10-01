"""CPU tests for negeig_sglang.py. No GPU, no SGLang install: the unpacked 0.5.20 source is used for the real
kernel files and for the real `Qwen3_5GatedDeltaNet.forward` text, everything else is a small stub.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python test_negeig_cpu.py --src /path/to/unpacked/sglang [--hf-patch patch.py]

What it covers: the producer kernel against a torch reference and against the HF patch; the two rewritten kernel
copies against the stock kernels (bit equality at W = 0, widened beta above 1 otherwise, packed decode against a
torch delta-rule step); the install wiring on stub classes (rewritten real forward, kernel dispatch by marker, backend
guards, load_weights tap, three gate modes, TP sharding, fp32 and shape refusals); the dist-info plugin discovery;
export-gates and merge-lora with red arms. What it cannot cover (CUDA graphs, real kernels on a GPU, TP > 1 processes)
is check_parity.py's job on the box.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import textwrap
import traceback
import types
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["NEGEIG_ALLOW_NON_CUDA"] = "1"

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import negeig_sglang as N  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def raises(name: str, exc, fn, match: str = "") -> None:
    try:
        fn()
    except exc as e:
        msg = str(e)
        check(name, match in msg, f"{type(e).__name__}: {msg[:140]}")
        return
    except BaseException as e:  # wrong type
        check(name, False, f"wrong exception {type(e).__name__}: {e}")
        return
    check(name, False, "no exception raised")


# ----------------------------------------------------------------------------------------------------------
# stub sglang
# ----------------------------------------------------------------------------------------------------------
def mod(name: str, path=None, **attrs):
    m = types.ModuleType(name)
    if path is not None:
        m.__path__ = [str(path)]
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def build_stub_packages(src: Path) -> None:
    mod("sglang", src, __version__=N.PINNED_SGLANG)
    mod("sglang.kernels", src / "kernels")
    mod("sglang.kernels.ops", src / "kernels/ops")
    mod("sglang.kernels.ops.attention", src / "kernels/ops/attention")
    mod("sglang.kernels.ops.attention.fla", src / "kernels/ops/attention/fla")
    mod("sglang.kernels.jit", src / "kernels/jit")
    mod("sglang.kernels.jit.utils", is_arch_support_pdl=lambda: False)
    mod("sglang.srt", src / "srt")
    mod("sglang.srt.utils", src / "srt/utils")
    from packaging import version as pv

    mod("sglang.srt.utils.common", torch_release=pv.parse(torch.__version__).release)
    if os.environ.get("TRITON_INTERPRET") == "1":
        # Interpreter-only workaround: fla.op binds `exp = tl.exp` at import, before the interpreter patches the
        # language module, so a kernel calling the bound name hits the JIT-only builtin. Resolve tl.exp at call time.
        import triton.language as tl

        op = importlib.import_module("sglang.kernels.ops.attention.fla.op")
        op.exp = lambda x: tl.exp(x)


STUB_QWEN = '''
from __future__ import annotations
import torch
import torch.nn as nn
import triton
from typing import Iterable, Set, Tuple

_is_xpu = False
_is_cpu = True
_is_npu = False
_is_cuda = False
_gdn_decode_fused_proj_conv = True
_GDN_FUSED_QKVZBA_RATIOS = (1, 2, 4)


def fused_qkvzba_split_reshape_cat_contiguous(pq, pb, nk, nv, dk, dv):
    T = pq.shape[0]
    q, k, v, z = torch.split(pq, [nk * dk, nk * dk, nv * dv, nv * dv], dim=-1)
    b, a = torch.split(pb, [nv, nv], dim=-1)
    mixed = torch.cat([q, k, v], dim=-1).contiguous()
    return mixed, z.reshape(T, nv, dv), b.contiguous(), a.contiguous()


qwen3_5_gdn_prefill_projection_views = fused_qkvzba_split_reshape_cat_contiguous


class _Attn(nn.Module):
    def __init__(self, A_log, nv, dv):
        super().__init__()
        self.A_log = A_log
        self.nv, self.dv = nv, dv
        self.calls = []

    def forward(self, forward_batch, mixed_qkv, a, b):
        self.calls.append({"mixed_qkv": mixed_qkv, "a": a, "b": b.clone()})
        out = torch.zeros(a.shape[0], self.nv, self.dv)
        if isinstance(mixed_qkv, tuple):  # the stock fused decode path returns (core_attn_out, z)
            return out, torch.zeros_like(out)
        return out


class _Id(nn.Module):
    def forward(self, x, z):
        return x


class _Out(nn.Module):
    def forward(self, x):
        return x, None


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, hidden=64, nk=2, nv=4, dk=16, dv=16, layer_id=0, tp=1, rank=0):
        super().__init__()
        self.attn_tp_rank, self.attn_tp_size = rank, tp
        self.hidden_size, self.num_v_heads, self.num_k_heads = hidden, nv, nk
        self.head_k_dim, self.head_v_dim, self.layer_id = dk, dv, layer_id
        nvt = nv // tp
        self.dt_bias = nn.Parameter(torch.zeros(nvt))
        self.A_log = nn.Parameter(torch.zeros(nvt))
        self.attn = _Attn(self.A_log, nvt, dv)
        self.qkvz_w = (2 * (nk // tp) * dk + 2 * nvt * dv)
        self.proj = nn.Linear(hidden, self.qkvz_w + 2 * nvt, bias=False)
        self.norm = _Id()
        self.out_proj = _Out()

    def _forward_input_proj(self, hidden_states):
        if isinstance(hidden_states, tuple):
            raise AssertionError("tuple reached the stub projection")
        y = self.proj(hidden_states.float()).to(hidden_states.dtype)
        return y[:, : self.qkvz_w], y[:, self.qkvz_w :]

FORWARD_TEXT

class _Model(nn.Module):
    def __init__(self, kinds, hidden, tp, rank, nv):
        super().__init__()
        self.embed = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList()
        for i, kind in enumerate(kinds):
            blk = nn.Module()
            if kind == "gdn":
                blk.linear_attn = Qwen3_5GatedDeltaNet(hidden=hidden, nv=nv, layer_id=i, tp=tp, rank=rank)
            else:
                blk.self_attn = nn.Linear(4, 4, bias=False)
            self.layers.append(blk)


def _load(self, weights):
    loaded: Set[str] = set()
    params = dict(self.named_parameters(remove_duplicate=False))
    for name, w in weights:
        name = name.replace("model.language_model.", "model.")
        if name not in params:
            continue
        p = params[name]
        getattr(p, "weight_loader", lambda p, w: p.data.copy_(w))(p, w)
        loaded.add(name)
    return loaded


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, kinds=("gdn", "attn", "gdn"), hidden=64, tp=1, rank=0, nv=4):
        super().__init__()
        self.model = _Model(kinds, hidden, tp, rank, nv)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return _load(self, weights)


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return super().load_weights(weights)


class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM):
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return _load(self, weights)


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForCausalLM):
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return _load(self, weights)
'''


def install_stub_backends(src: Path, tmp: Path):
    """Stub qwen module (real forward text), stub gdn_backend and gdn_triton around the real fla kernel files."""
    qwen_real = (src / N.REL_QWEN).read_text()
    fwd = N._forward_segment(qwen_real)
    fwd = textwrap.indent(fwd, "    ")
    stub_src = STUB_QWEN.replace("FORWARD_TEXT", fwd)
    path = tmp / "stub_qwen3_5.py"
    path.write_text(stub_src)
    spec = importlib.util.spec_from_file_location(N.MOD_QWEN, path)
    M = importlib.util.module_from_spec(spec)
    sys.modules[N.MOD_QWEN] = M
    spec.loader.exec_module(M)

    gating = importlib.import_module(N.MOD_GATING)
    recurrent = importlib.import_module(N.MOD_RECURRENT)

    class TritonGDNKernel:
        pass

    class OtherKernel:
        pass

    def _stock_generic(*a, **k):
        return "stock-generic"

    mod(
        N.MOD_GT,
        TritonGDNKernel=TritonGDNKernel,
        fused_recurrent_gated_delta_rule_packed_decode=recurrent.fused_recurrent_gated_delta_rule_packed_decode,
        fused_recurrent_gdn_replayssm_decode=lambda *a, **k: "stock-replayssm",
        fused_sigmoid_gating_delta_rule_update=_stock_generic,
    )
    mod("sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent", fused_sigmoid_gating_delta_rule_update=_stock_generic)

    class Mode:
        def __init__(self, tv):
            self.tv = tv

        def is_target_verify(self):
            return self.tv

    class GDNAttnBackend:
        def __init__(self, decode_kernel, extend_kernel):
            self.kernel_dispatcher = types.SimpleNamespace(decode_kernel=decode_kernel, extend_kernel=extend_kernel)
            self.calls = []

        def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kw):
            self.calls.append(("decode", b))
            return "decode-ok"

        def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kw):
            self.calls.append(("extend", b))
            return "extend-ok"

    mod(N.MOD_GB, fused_gdn_gating=gating.fused_gdn_gating, GDNAttnBackend=GDNAttnBackend)
    return M, TritonGDNKernel, OtherKernel, Mode, GDNAttnBackend


class FB:
    """Minimal forward batch."""

    def __init__(self, decode: bool, tv: bool = False):
        self.batch_size = 1
        self.forward_mode = types.SimpleNamespace(
            is_decode=lambda: decode, is_target_verify=lambda: tv, is_extend_without_speculative=lambda: not decode
        )


# ----------------------------------------------------------------------------------------------------------
def rtz_bf16(x: torch.Tensor) -> torch.Tensor:
    """float32 -> bfloat16 rounding toward zero (what the Triton CPU interpreter does; the GPU rounds to nearest)."""
    return (x.float().contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


def beta_reference_rtz(b, t):
    s = rtz_bf16(torch.sigmoid(b.float())).float()
    coef = torch.where(t >= 0, 2.0 - s, s)
    return rtz_bf16(s + t.float() * coef)


def test_pure(hf_patch: Path):
    torch.manual_seed(0)
    T, NV, H = 37, 48, 128
    b = (torch.randn(T, NV) * 3).to(torch.bfloat16)
    x = torch.randn(T, H).to(torch.bfloat16)
    w0 = torch.zeros(NV, H)
    w = torch.randn(NV, H) * 0.4
    stock = torch.sigmoid(b.float()).to(torch.bfloat16)  # what the stock bf16 path yields on any device

    # torch path (the production fallback, and the arithmetic the GPU kernel mirrors): bit-exact claims live here
    real_ok = N._triton_ok
    N._triton_ok = lambda b_: False
    try:
        t0 = N.gate_t(x, w0)
        out0 = N.beta_combine(b, t0)
        bad = int((out0 != stock).sum())
        check("torch path: W=0 beta equals the stock bf16 sigmoid, bit for bit", bad == 0, f"{bad} of {out0.numel()} differ")
        check("torch path: t at W=0 is +0 (so the coefficient is 2 - s and t * coef is 0)", bool((t0 == 0).all()) and not bool(torch.signbit(t0).any()))
        t = N.gate_t(x, w)
        got = N.beta_combine(b, t)
        check("torch path: beta_combine equals beta_reference", torch.equal(got, N.beta_reference(b, t)))
        check("non-vacuity: widened beta exceeds 1 and differs from stock", bool((got.float() > 1).any()) and not torch.equal(got, stock), f"max beta {float(got.float().max()):.3f}, min {float(got.float().min()):.3f}")
        check("beta stays inside [0, 2]", bool(((got.float() >= 0) & (got.float() <= 2)).all()))
        if hf_patch.is_file():
            spec = importlib.util.spec_from_file_location("hf_negeig_patch", hf_patch)
            P = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(P)
            lay = types.SimpleNamespace(negeig_w=nn.Linear(H, NV, bias=False).float())
            with torch.no_grad():
                lay.negeig_w.weight.copy_(w)
            hf = P._negeig_beta(lay, b, x)
            bad = int((hf != got).sum())
            check("torch path equals the HF patch _negeig_beta (bf16 exact)", bad == 0, f"{bad} of {hf.numel()} differ")
            with torch.no_grad():
                lay.negeig_w.weight.zero_()
            hf0 = P._negeig_beta(lay, b, x)
            check("HF patch at W=0 equals the stock bf16 sigmoid", torch.equal(hf0, stock))
            check("torch path at W=0 equals the HF patch at W=0", torch.equal(hf0, out0))
        else:
            check("HF patch comparison", False, f"{hf_patch} not found")
    finally:
        N._triton_ok = real_ok

    # Triton kernel under the CPU interpreter (rounds toward zero, so compare with an RTZ twin of the reference)
    t = N.gate_t(x, w)
    got_k = N.beta_combine(b, t)
    ref_k = beta_reference_rtz(b, t)
    diff = (got_k.float() - ref_k.float()).abs()
    ulp_bad = int((diff > 2.0 ** -7 * torch.maximum(got_k.float().abs(), ref_k.float().abs()).clamp_min(1e-30)).sum())
    eq = int((got_k == ref_k).sum())
    check("interpreter kernel matches the RTZ twin of the reference (within 1 ulp, >99% exact)", ulp_bad == 0 and eq >= 0.99 * got_k.numel(), f"{eq} of {got_k.numel()} exact, {ulp_bad} beyond 1 ulp")
    check("interpreter kernel agrees with the torch path within 1 ulp-ish", bool(((got_k.float() - N.beta_reference(b, t).float()).abs() <= 0.02).all()))
    got_k0 = N.beta_combine(b, N.gate_t(x, w0))
    check("interpreter kernel at W=0 is the sigmoid to within the interpreter's RTZ", bool(((got_k0.float() - stock.float()).abs() <= 2.0 ** -7 * stock.float().abs().clamp_min(1e-30)).all()) and torch.equal(got_k0, rtz_bf16(torch.sigmoid(b.float()))))

    # strided b (a view of a wider buffer) and a new contiguous output
    big = torch.randn(T, NV + 8).to(torch.bfloat16)
    view = big[:, :NV]
    got_v = N.beta_combine(view, t)
    check("strided b input handled (same result as the contiguous copy)", torch.equal(got_v, N.beta_combine(view.contiguous(), t)) and got_v.is_contiguous() and got_v.data_ptr() != view.data_ptr())
    empty = N.beta_combine(b[:0], t[:0])
    check("zero rows handled", empty.shape == (0, NV))
    raises("shape mismatch refused", N.NegeigError, lambda: N.beta_combine(b, t[:, :4]), "shape mismatch")
    raises("non-fp32 W refused", N.NegeigError, lambda: N.beta_eager(b, x, w.to(torch.bfloat16)), "fp32")

    # row chunking (fp32 summation order differs between a chunked and a single call: tiny, but real)
    a = N.gate_t(x, w, chunk=N.ROW_CHUNK)
    c = N.gate_t(x, w, chunk=7)
    check("row-chunked fp32 gate equals single call within fp32 summation noise", torch.allclose(a, c, atol=1e-5), f"max diff {float((a - c).abs().max()):.2e}, bitwise equal: {torch.equal(a, c)}")

    # precision context restores the setting and really runs at highest
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        N.gate_t(x, w)
        restored = torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(prev)
    check("float32 matmul precision restored after the gate matmul", restored)


def test_sources(src: Path):
    qwen = (src / N.REL_QWEN).read_text()
    seg = N._forward_segment(qwen)
    new = N.rewrite_forward_source(seg)
    check("forward rewrite adds exactly the producer and the gate flag", new.count("\n") == seg.count("\n") + 4, f"lines {seg.count(chr(10)) + 1} -> {new.count(chr(10))}")
    check("forward rewrite compiles", bool(compile(new, "<t>", "exec")))
    raises("forward rewrite refuses a changed call line", N.NegeigPatchError, lambda: N.rewrite_forward_source(seg.replace("attn_result = self.attn(", "out = self.attn(")), "targets changed")
    raises("forward rewrite refuses a duplicated flag line", N.NegeigPatchError, lambda: N.rewrite_forward_source(seg + "\n    _gdn_decode_fused_proj_conv\n"), "targets changed")
    g = (src / N.REL_GATING).read_text()
    r = (src / N.REL_RECURRENT).read_text()
    raises("kernel rewrite refuses a missing target", N.NegeigPatchError, lambda: N.rewrite_kernel_source(g.replace(N.GATING_OLD, "pass"), N.GATING_OLD, N.GATING_NEW, "g"), "0 times")
    raises("kernel rewrite refuses a duplicated target", N.NegeigPatchError, lambda: N.rewrite_kernel_source(r + "\n" + N.PACKED_OLD, N.PACKED_OLD, N.PACKED_NEW, "r"), "2 times")
    res = N.static_check(str(src))
    check("static-check on the pinned source", res["ok"] and res["sglang_version"] == N.PINNED_SGLANG, json.dumps(res)[:160])
    check("KDA packed-decode sigmoid line left untouched", N.rewrite_kernel_source(r, N.PACKED_OLD, N.PACKED_NEW, "r").count("tl.sigmoid(b_val)") == r.count("tl.sigmoid(b_val)") - 1)


def test_kernels(src: Path, tmp: Path):
    gating = importlib.import_module(N.MOD_GATING)
    rec = importlib.import_module(N.MOD_RECURRENT)
    pg = N.patched_kernel_module(gating, N.GATING_OLD, N.GATING_NEW, "fused_gdn_gating")
    pr = N.patched_kernel_module(rec, N.PACKED_OLD, N.PACKED_NEW, "fused_recurrent")
    check("patched kernel copies are real files", Path(pg.__file__).is_file() and Path(pr.__file__).is_file(), Path(pg.__file__).name)
    # the same content loads to the same module
    pg2 = N.patched_kernel_module(gating, N.GATING_OLD, N.GATING_NEW, "fused_gdn_gating")
    check("patched kernel copy is cached by content", pg2 is pg)

    torch.manual_seed(1)
    T, NV, H = 19, 8, 32
    A_log = torch.randn(NV)
    dt_bias = torch.randn(NV)
    a = torch.randn(T, NV).to(torch.bfloat16)
    b = (torch.randn(T, NV) * 3).to(torch.bfloat16)
    g_s, beta_s = gating.fused_gdn_gating(A_log, a, b, dt_bias)
    x = torch.randn(T, H).to(torch.bfloat16)
    b_final0 = N.beta_combine(b, N.gate_t(x, torch.zeros(NV, H)))
    g_p, beta_p = pg.fused_gdn_gating(A_log, a, b_final0, dt_bias)
    check("prefill gating copy: beta bit-equal to stock at W=0", torch.equal(beta_s, beta_p), f"max diff {float((beta_s - beta_p).abs().max())}")
    check("prefill gating copy: g bit-equal to stock", torch.equal(g_s, g_p))
    w = torch.randn(NV, H) * 0.6
    b_final = N.beta_combine(b, N.gate_t(x, w))
    g_w, beta_w = pg.fused_gdn_gating(A_log, a, b_final, dt_bias)
    check("prefill gating copy: passes the final beta through unchanged", torch.equal(beta_w.reshape(T, NV), b_final.float()), f"max beta {float(beta_w.max()):.3f}")
    check("prefill gating copy: beta above 1 survives", bool((beta_w > 1).any()))
    _, beta_bad = gating.fused_gdn_gating(A_log, a, b_final, dt_bias)
    check("red arm: the STOCK kernel re-applies the sigmoid to a final beta", not torch.allclose(beta_bad.reshape(T, NV), b_final.float()) and bool((beta_bad <= 1).all()))

    # packed decode: stock vs copy at W = 0, copy vs a torch delta-rule step otherwise
    B, Hk, HV, K, V = 3, 2, 4, 16, 16
    qkv_dim = 2 * Hk * K + HV * V
    mixed = torch.randn(B, qkv_dim).to(torch.bfloat16)
    a2 = torch.randn(B, HV).to(torch.bfloat16)
    b2 = (torch.randn(B, HV) * 2).to(torch.bfloat16)
    A2 = torch.randn(HV)
    dt2 = torch.randn(HV)
    slots = 5
    state0 = torch.randn(slots, HV, V, K) * 0.2
    idx = torch.tensor([3, 0, 4], dtype=torch.int32)
    scale = K**-0.5

    def run(fn, bb):
        st = state0.clone()
        out = torch.empty(B, 1, HV, V, dtype=torch.bfloat16)
        fn(mixed_qkv=mixed, a=a2, b=bb, A_log=A2, dt_bias=dt2, scale=scale, initial_state=st, out=out, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
        return out, st

    x2 = torch.randn(B, H).to(torch.bfloat16)
    b2_final0 = N.beta_combine(b2, N.gate_t(x2, torch.zeros(HV, H)))
    o_s, s_s = run(rec.fused_recurrent_gated_delta_rule_packed_decode, b2)
    o_p, s_p = run(pr.fused_recurrent_gated_delta_rule_packed_decode, b2_final0)
    check("packed decode copy: output and state bit-equal to stock at W=0", torch.equal(o_s, o_p) and torch.equal(s_s, s_p))

    w2 = torch.randn(HV, H) * 1.5
    b2_final = N.beta_combine(b2, N.gate_t(x2, w2))
    o_w, s_w = run(pr.fused_recurrent_gated_delta_rule_packed_decode, b2_final)

    def ref_step(beta_bb):
        outs, st = [], state0.clone()
        for i in range(B):
            q, k, v = torch.split(mixed[i].float(), [Hk * K, Hk * K, HV * V])
            for hv in range(HV):
                hk = hv // (HV // Hk)
                qq, kk = q[hk * K : (hk + 1) * K], k[hk * K : (hk + 1) * K]
                qq = qq / torch.sqrt((qq * qq).sum() + 1e-6) * scale
                kk = kk / torch.sqrt((kk * kk).sum() + 1e-6)
                vv = v[hv * V : (hv + 1) * V]
                h = st[idx[i], hv]
                xx = a2[i, hv].float() + dt2[hv]
                sp = torch.where(xx <= 20.0, torch.log1p(torch.exp(xx)), xx)
                gg = -torch.exp(A2[hv]) * sp
                h = h * torch.exp(gg)
                dv_ = (vv - h @ kk) * beta_bb[i, hv].float()
                h = h + dv_[:, None] * kk[None, :]
                st[idx[i], hv] = h
                outs.append(h @ qq)
        return torch.stack(outs).reshape(B, 1, HV, V), st

    o_ref, s_ref = ref_step(b2_final)
    check("packed decode copy: beta above 1 equals a torch delta-rule step", torch.allclose(o_w.float(), o_ref, atol=2e-2, rtol=2e-2) and torch.allclose(s_w, s_ref, atol=1e-4, rtol=1e-4),
          f"state max err {float((s_w - s_ref).abs().max()):.2e}, beta max {float(b2_final.float().max()):.2f}")
    check("packed decode non-vacuity: widened beta changes the state", not torch.equal(s_w, s_s))
    _, s_bad = run(rec.fused_recurrent_gated_delta_rule_packed_decode, b2_final)
    check("red arm: the STOCK decode kernel given a final beta differs from the copy", not torch.equal(s_bad, s_w))


def test_wiring(src: Path, tmp: Path):
    # bit-exact wiring claims use the torch producer path (the interpreter rounds f32 to bf16 toward zero)
    real_ok = N._triton_ok
    N._triton_ok = lambda b_: False
    try:
        _test_wiring(src, tmp)
    finally:
        N._triton_ok = real_ok


def _test_wiring(src: Path, tmp: Path):
    M, TritonK, OtherK, Mode, Backend = install_stub_backends(src, tmp)
    os.environ[N.ENV_CACHE_DIR] = str(tmp / "cache")
    raises("register() without NEGEIG_GATES is a SystemExit, not an Exception", SystemExit, lambda: (os.environ.pop(N.ENV_GATES, None), N.register()), "NEGEIG_GATES")
    check("SystemExit subclass escapes a plugin loader's except Exception", issubclass(N.NegeigPatchError, SystemExit) and not issubclass(N.NegeigPatchError, Exception))
    sys.modules["sglang"].__version__ = "0.5.21"
    os.environ[N.ENV_GATES] = "zeros"
    raises("other sglang version refused", N.NegeigPatchError, N.install, "pinned")
    sys.modules["sglang"].__version__ = N.PINNED_SGLANG

    N.install()
    N.install()  # idempotent
    check("install is idempotent", getattr(M.Qwen3_5GatedDeltaNet.forward, "_negeig_wrapped", False) and not hasattr(M.Qwen3_5GatedDeltaNet.forward, "__wrapped__"))
    check("rewritten forward lives in a real cache file", Path(M.Qwen3_5GatedDeltaNet.forward.__code__.co_filename).is_file())

    # ---- layer construction
    layer = M.Qwen3_5GatedDeltaNet(hidden=64, nv=4, layer_id=0)
    check("layer has fp32 zero W [nv, hidden]", layer.negeig_w.dtype == torch.float32 and tuple(layer.negeig_w.shape) == (4, 64) and float(layer.negeig_w.abs().sum()) == 0.0 and not layer.negeig_w.requires_grad)
    check("marker on the shared A_log Parameter", getattr(layer.attn.A_log, "_negeig", False) is True and layer.attn.A_log is layer.A_log)
    layer_tp = M.Qwen3_5GatedDeltaNet(hidden=64, nv=4, layer_id=0, tp=2, rank=1)
    check("TP shard shape [nv/tp, hidden]", tuple(layer_tp.negeig_w.shape) == (2, 64))

    # ---- forward: the attention backend receives the final beta
    torch.manual_seed(2)
    x = torch.randn(9, 64).to(torch.bfloat16)
    b_ref = layer._forward_input_proj(x)[1][:, :4].contiguous()
    layer(x, FB(decode=False))
    got0 = layer.attn.calls[-1]["b"]
    check("forward at W=0 hands the backend bf16 sigmoid(b), bit-equal", torch.equal(got0, torch.sigmoid(b_ref.float()).to(torch.bfloat16)))
    w = torch.randn(4, 64) * 0.5
    layer.negeig_w.data.copy_(w)
    layer(x, FB(decode=False))
    got = layer.attn.calls[-1]["b"]
    check("forward with random W hands the backend the widened beta", torch.equal(got, N.beta_eager(b_ref, x, w)) and bool((got.float() > 1).any()))
    layer(x[:1], FB(decode=True))
    calls = layer.attn.calls[-1]
    check("decode: fused projection path is off for a gated layer (mixed_qkv is a tensor)", isinstance(calls["mixed_qkv"], torch.Tensor))
    dec_ref = N.beta_eager(layer._forward_input_proj(x[:1])[1][:, :4].contiguous(), x[:1], w)
    check("decode beta equals prefill beta for the same token", torch.equal(calls["b"], dec_ref) and torch.equal(calls["b"], got[:1]), f"row0 diff {float((calls['b'].float() - got[:1].float()).abs().max())}")
    # SGLang's tc_piecewise backend runs torch.compile(fullgraph=True) over the decoder forward. The producer must be
    # traceable there: a lock or any other untraceable call inside it (the first version took _OP_LOCK per call) fails.
    import torch._dynamo as dynamo

    dynamo.reset()
    b_fg = layer._forward_input_proj(x)[1][:, :4].contiguous()
    fg = torch.compile(lambda xx, bb: layer._negeig_apply(xx, bb), fullgraph=True, backend="eager")
    check("producer traces under torch.compile(fullgraph=True) and equals eager", torch.equal(fg(x, b_fg), N.beta_eager(b_fg, x, layer.negeig_w.data)))
    dynamo.reset()
    fb_fg = FB(decode=False)
    fg_layer = torch.compile(lambda xx: layer(xx, fb_fg), fullgraph=True, backend="eager")
    try:
        fg_layer(x)
        layer_fg_ok, why = True, ""
    except Exception as e:  # stub attention backends may be untraceable; the producer call itself is covered above
        layer_fg_ok, why = False, f"{type(e).__name__}: {str(e)[:160]}"
    if layer_fg_ok:
        check("rewritten forward traces under fullgraph and hands the backend the widened beta", torch.equal(layer.attn.calls[-1]["b"], got))
    else:
        print("note  fullgraph over the whole stub forward not traceable (" + why + "); producer traced alone above", flush=True)
    dynamo.reset()
    layer.negeig_w.data = layer.negeig_w.data.to(torch.bfloat16)
    raises("bf16 W at runtime refused", N.NegeigError, lambda: layer(x, FB(decode=False)), "fp32")
    layer.negeig_w.data = layer.negeig_w.data.float()
    layer._negeig = False
    layer(x[:1], FB(decode=True))
    check("an ungated layer keeps the stock fused decode path (mixed_qkv tuple)", isinstance(layer.attn.calls[-1]["mixed_qkv"], tuple))
    layer._negeig = True

    # ---- kernel dispatch by marker
    GB = sys.modules[N.MOD_GB]
    GT = sys.modules[N.MOD_GT]
    gating = importlib.import_module(N.MOD_GATING)
    A_log = torch.randn(4)
    a = torch.randn(5, 4).to(torch.bfloat16)
    bf = (torch.randn(5, 4) * 2).to(torch.bfloat16)
    dtb = torch.randn(4)
    plain_A = nn.Parameter(A_log.clone())
    _, beta_plain = GB.fused_gdn_gating(plain_A, a, bf, dtb)
    _, beta_stock = gating.fused_gdn_gating(plain_A, a, bf, dtb)
    check("unmarked A_log: the stock kernel runs (sigmoid kept, bf16-rounded)", torch.equal(beta_plain, beta_stock) and bool((beta_plain <= 1).all()) and not torch.equal(beta_plain.reshape(5, 4), bf.float()))
    marked_A = nn.Parameter(A_log.clone())
    marked_A._negeig = True
    _, beta_m = GB.fused_gdn_gating(marked_A, a, bf, dtb)
    check("marked A_log: passthrough (final beta unchanged)", torch.equal(beta_m.reshape(5, 4), bf.float()))
    check("gating wrapper keeps the stock signature", GB.fused_gdn_gating.__wrapped__ is gating.fused_gdn_gating)
    raises("generic fused sigmoid decode refused for a gated layer", N.NegeigError, lambda: GT.fused_sigmoid_gating_delta_rule_update(A_log=marked_A), "cannot take the widened beta")
    check("generic fused sigmoid decode untouched for a stock layer", GT.fused_sigmoid_gating_delta_rule_update(A_log=plain_A) == "stock-generic")
    raises("replay-SSM decode refused for a gated layer", N.NegeigError, lambda: GT.fused_recurrent_gdn_replayssm_decode(A_log=marked_A), "replay-SSM")
    sig = sys.modules["sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent"]
    raises("target-verify helper kernel refused for a gated layer", N.NegeigError, lambda: sig.fused_sigmoid_gating_delta_rule_update(A_log=marked_A), "cannot take")

    # ---- backend guards
    be = Backend(TritonK(), TritonK())
    be_bad = Backend(OtherK(), OtherK())
    lay = types.SimpleNamespace(A_log=marked_A)
    stock_lay = types.SimpleNamespace(A_log=plain_A)
    check("decode guard passes a gated layer on Triton", be.forward_decode(lay, FB(True), torch.zeros(2, 2), a, bf) == "decode-ok")
    raises("decode guard refuses a tuple mixed_qkv", N.NegeigError, lambda: be.forward_decode(lay, FB(True), (1, 2), a, bf), "tuple")
    raises("decode guard refuses a non-Triton decode kernel", N.NegeigError, lambda: be_bad.forward_decode(lay, FB(True), torch.zeros(2, 2), a, bf), "triton")
    check("decode guard leaves a stock layer alone", be_bad.forward_decode(stock_lay, FB(True), (1, 2), a, bf) == "decode-ok")
    check("extend guard passes a gated layer on Triton", be.forward_extend(lay, FB(False), torch.zeros(2, 2), a, bf) == "extend-ok")
    raises("extend guard refuses target verify", N.NegeigError, lambda: be.forward_extend(lay, FB(False, tv=True), torch.zeros(2, 2), a, bf), "target verify")
    raises("extend guard refuses a non-Triton prefill kernel", N.NegeigError, lambda: be_bad.forward_extend(lay, FB(False), torch.zeros(2, 2), a, bf), "triton")
    os.environ[N.ENV_ALLOW_FLASHINFER_PREFILL] = "1"
    check("extend guard override env lets a non-Triton prefill through", be_bad.forward_extend(lay, FB(False), torch.zeros(2, 2), a, bf) == "extend-ok")
    os.environ.pop(N.ENV_ALLOW_FLASHINFER_PREFILL)
    check("extend guard leaves a stock layer alone", be_bad.forward_extend(stock_lay, FB(False, tv=True), torch.zeros(2, 2), a, bf) == "extend-ok")

    # ---- load_weights: tap, file, zeros, stream, TP, refusals
    from safetensors.torch import save_file

    def gate_file(ids, shape=(4, 64), dtype=torch.float32, key="model.language_model.layers.{}.linear_attn.negeig_w"):
        d = {key.format(i): torch.randn(*shape).to(dtype) * 0.3 for i in ids}
        p = tmp / f"gates_{len(list(tmp.iterdir()))}.safetensors"
        save_file(d, str(p))
        return p, d

    def fresh(cls=M.Qwen3_5ForCausalLM, **kw):
        m = cls(**kw)
        for p in m.parameters():
            p.data.fill_(0.123)  # the capture-safe loader's finite sentinel
        return m

    ckpt = lambda: [("model.embed.weight", torch.ones(4, 4))]  # noqa: E731

    path, d = gate_file([0, 2])
    os.environ[N.ENV_GATES] = str(path)
    m = fresh()
    loaded = m.load_weights(iter(ckpt()))
    g0 = m.model.layers[0].linear_attn.negeig_w
    check("file mode: gates loaded into every GDN layer (attention layer 1 skipped)", torch.equal(g0, d["model.language_model.layers.0.linear_attn.negeig_w"]) and torch.equal(m.model.layers[2].linear_attn.negeig_w, d["model.language_model.layers.2.linear_attn.negeig_w"]))
    check("file mode: loaded-params set reports the gates", "model.layers.2.linear_attn.negeig_w" in loaded)
    check("file mode: the stock weight still loaded", torch.equal(m.model.embed.weight, torch.ones(4, 4)))
    m.model.layers[0].linear_attn.negeig_w.data.zero_()
    m.load_weights(iter(ckpt()))
    check("file mode: once per model instance (a later sync does not reapply the file)", float(m.model.layers[0].linear_attn.negeig_w.abs().sum()) == 0.0)

    path_miss, _ = gate_file([0])
    os.environ[N.ENV_GATES] = str(path_miss)
    raises("file missing a GDN layer refused", N.NegeigError, lambda: fresh().load_weights(iter(ckpt())), "layers [2]")
    path_bad, _ = gate_file([0, 2], shape=(3, 64))
    os.environ[N.ENV_GATES] = str(path_bad)
    raises("wrong gate shape refused", N.NegeigError, lambda: fresh().load_weights(iter(ckpt())), "expected the full")
    path_bf, _ = gate_file([0, 2], dtype=torch.bfloat16)
    os.environ[N.ENV_GATES] = str(path_bf)
    raises("bf16 gate file refused", N.NegeigError, lambda: fresh().load_weights(iter(ckpt())), "fp32")
    os.environ[N.ENV_ALLOW_LOWP_W] = "1"
    check("bf16 gate file accepted only with the override", fresh().load_weights(iter(ckpt())) is not None)
    os.environ.pop(N.ENV_ALLOW_LOWP_W)
    os.environ[N.ENV_GATES] = str(tmp / "nope.safetensors")
    raises("missing gate file refused", N.NegeigError, lambda: fresh().load_weights(iter(ckpt())), "neither")

    os.environ[N.ENV_GATES] = "zeros"
    m = fresh()
    m.load_weights(iter(ckpt()))
    check("zeros mode overwrites the capture sentinel with zero", all(float(mm.linear_attn.negeig_w.abs().sum()) == 0.0 for mm in m.model.layers if hasattr(mm, "linear_attn")))

    os.environ[N.ENV_GATES] = "stream"
    raises("stream mode without gates in the stream refused", N.NegeigError, lambda: fresh().load_weights(iter(ckpt())), "weight stream")
    full = {i: torch.randn(4, 64) * 0.3 for i in (0, 2)}
    stream = ckpt() + [(f"model.language_model.layers.{i}.linear_attn.negeig_w.weight", full[i]) for i in full]
    m = fresh()
    m.load_weights(iter(stream))
    check("stream mode: trainer key form (.negeig_w.weight, language_model prefix) reaches the param", all(torch.equal(m.model.layers[i].linear_attn.negeig_w, full[i]) for i in full))
    stream2 = ckpt() + [(f"model.layers.{i}.linear_attn.negeig_w", full[i] * 2) for i in full]
    m.load_weights(iter(stream2))
    check("a later weight sync updates the gate in place", torch.equal(m.model.layers[2].linear_attn.negeig_w, full[2] * 2))
    bf_stream = ckpt() + [(f"model.layers.{i}.linear_attn.negeig_w", full[i].to(torch.bfloat16)) for i in full]
    raises("bf16 gate in the weight stream refused (fp32 end to end)", N.NegeigError, lambda: fresh().load_weights(iter(bf_stream)), "fp32")

    # path wins over stream keys
    os.environ[N.ENV_GATES] = str(path)
    m = fresh()
    m.load_weights(iter(stream))
    check("file mode: the file wins over stream keys", torch.equal(m.model.layers[0].linear_attn.negeig_w, d["model.language_model.layers.0.linear_attn.negeig_w"]))

    # TP shard
    os.environ[N.ENV_GATES] = str(path)
    m = fresh(tp=2, rank=1)
    m.load_weights(iter(ckpt()))
    check("TP rank 1 holds value-head rows 2:4 of the full gate", torch.equal(m.model.layers[0].linear_attn.negeig_w, d["model.language_model.layers.0.linear_attn.negeig_w"][2:4]))
    m0 = fresh(tp=2, rank=0)
    m0.load_weights(iter(ckpt()))
    check("TP rank 0 holds rows 0:2", torch.equal(m0.model.layers[0].linear_attn.negeig_w, d["model.language_model.layers.0.linear_attn.negeig_w"][:2]))

    # nesting: Moe -> CausalLM wrapper acts once; all four classes patched
    calls = []
    orig_after = N._after_load
    N._after_load = lambda model, MM: (calls.append(type(model).__name__), orig_after(model, MM))[1]
    try:
        for cls in (M.Qwen3_5ForCausalLM, M.Qwen3_5MoeForCausalLM, M.Qwen3_5ForConditionalGeneration, M.Qwen3_5MoeForConditionalGeneration):
            calls.clear()
            mm = fresh(cls)
            mm.load_weights(iter(ckpt()))
            check(f"{cls.__name__}: gates applied exactly once", calls == [cls.__name__], str(calls))
    finally:
        N._after_load = orig_after
    check("all four load_weights wrapped", all(getattr(getattr(M, c).load_weights, "_negeig_wrapped", False) for c in ("Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration")))

    # model with no gated layer
    none_model = M.Qwen3_5ForCausalLM(kinds=("attn", "attn"))
    raises("a model with no gated GDN layer is refused", N.NegeigError, lambda: none_model.load_weights(iter(ckpt())), "no gated")


def test_plugin(tmp: Path):
    d = tmp / "plug"
    info = N.install_plugin(str(d))
    sys.path.insert(0, str(d))
    try:
        eps = importlib.metadata.entry_points(group="sglang.srt.plugins")
        names = {e.name: e for e in eps}
        check("dist-info entry point discoverable without pip", "negeig" in names and names["negeig"].value == "negeig_sglang:register", json.dumps(info["launch_env"])[:100])
        check("the plugin copy is byte-identical to the source", (d / "negeig_sglang.py").read_bytes() == Path(N.__file__).read_bytes())
    finally:
        sys.path.remove(str(d))
        importlib.invalidate_caches()


def test_export(tmp: Path):
    torch.manual_seed(3)
    cfg = {"text_config": {"layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"]}}
    cfgp = tmp / "config.json"
    cfgp.write_text(json.dumps(cfg))
    state = {}
    for i in (0, 1, 3):
        state[f"base_model.model.model.layers.{i}.linear_attn.negeig_w.weight"] = torch.randn(4, 64) * 0.2
        state[f"base_model.model.model.layers.{i}.linear_attn.in_proj_b.lora_A.default.weight"] = torch.randn(2, 64)
    p = tmp / "trainable_000010.pt"
    torch.save(state, p)
    out = tmp / "gates.safetensors"
    rec = N.export_gates(str(p), str(out), str(cfgp))
    back = N.read_gates_file(str(out))
    check("export-gates round trip (peft-prefixed trainer keys to fp32 file)", sorted(back) == [0, 1, 3] and all(v.dtype == torch.float32 and v.shape == (4, 64) for v in back.values()) and rec["n_layers"] == "3")
    check("export-gates values identical", torch.equal(back[1], state["base_model.model.model.layers.1.linear_attn.negeig_w.weight"]))
    p2 = tmp / "ckpt_000010.pt"
    torch.save({"trainable": state, "step": 10}, p2)
    check("export-gates reads a ckpt with a 'trainable' key", N.export_gates(str(p2), str(tmp / "g2.safetensors"), str(cfgp))["n_layers"] == "3")
    bad = dict(state)
    k0 = "base_model.model.model.layers.3.linear_attn.negeig_w.weight"
    del bad[k0]
    torch.save(bad, tmp / "miss.pt")
    raises("export-gates refuses a layer set that differs from the config", SystemExit, lambda: N.export_gates(str(tmp / "miss.pt"), str(tmp / "g3.safetensors"), str(cfgp)), "differ")
    nan = dict(state)
    nan[k0] = torch.full((4, 64), float("nan"))
    torch.save(nan, tmp / "nan.pt")
    raises("export-gates refuses non-finite gates", SystemExit, lambda: N.export_gates(str(tmp / "nan.pt"), str(tmp / "g4.safetensors")), "non-finite")
    wa = dict(state)
    wa["base_model.model.model.layers.0.linear_attn.negeig_wa.weight"] = torch.zeros(4, 64)
    torch.save(wa, tmp / "wa.pt")
    raises("export-gates refuses a decay-gate checkpoint", SystemExit, lambda: N.export_gates(str(tmp / "wa.pt"), str(tmp / "g5.safetensors")), "decay-gate")


def test_merge(tinyqwen: Path, tmp: Path):
    from safetensors import safe_open

    if not (tinyqwen / "model.safetensors").is_file():
        check("merge-lora on tinyqwen", False, f"{tinyqwen} missing")
        return
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    targets = r".*layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_a|in_proj_b|out_proj)"
    model = AutoModelForCausalLM.from_pretrained(str(tinyqwen), dtype=torch.bfloat16)
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=targets, bias="none"))
    torch.manual_seed(4)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.05)
    trainable = {n: p.detach().float().cpu() for n, p in pm.named_parameters() if p.requires_grad}
    check("peft trainable keys parse", len(trainable) > 0 and sum(1 for k in trainable if N.LORA_KEY.match(k)) == len(trainable), next(iter(trainable)))
    tp = tmp / "trainable_lora.pt"
    torch.save(trainable, tp)
    merged_peft = pm.merge_and_unload()
    outd = tmp / "merged"
    rec = N.merge_lora(str(tinyqwen), str(tp), str(outd), scale=2.0, rounding="stochastic", seed=7, min_delta_kept=0.9)
    check("merge-lora reports delta_kept near 1 with stochastic rounding", rec["delta_kept"] > 0.97, f"{rec['delta_kept']:.5f}, worst {rec['delta_kept_worst_tensor']:.4f}, {rec['targets']} tensors")
    pd = dict(merged_peft.named_parameters())
    worst = 0.0
    n_cmp = 0
    with safe_open(str(outd / "model.safetensors"), framework="pt") as f, safe_open(str(tinyqwen / "model.safetensors"), framework="pt") as f0:
        for k in f.keys():  # noqa: SIM118
            t = f.get_tensor(k)
            cand = [pk for pk in pd if pk.endswith(k.replace("model.language_model.", "model.").replace("model.layers", "layers"))]
            if "linear_attn" in k and k.endswith(("in_proj_qkv.weight", "in_proj_a.weight", "in_proj_b.weight", "out_proj.weight")) and cand:
                want = pd[cand[0]].detach().float()
                delta = (want - f0.get_tensor(k).float()).abs().max().item()
                err = (t.float() - want).abs().max().item()
                worst = max(worst, err / max(delta, 1e-9))
                n_cmp += 1
    check("merged tensors agree with peft merge_and_unload within bf16 rounding", n_cmp > 0 and worst < 0.5, f"{n_cmp} tensors, worst err/delta {worst:.3f}")
    check("merge receipt written", (outd / "MERGE-RECEIPT.json").is_file())
    # the rounding trap: a tiny RL-sized delta under round-to-nearest keeps almost nothing
    tiny = {k: (v * 0.0005 if "lora_B" in k else v) for k, v in trainable.items()}
    tpt = tmp / "trainable_tiny.pt"
    torch.save(tiny, tpt)
    raises("red arm: round-to-nearest merge of a tiny delta is refused", SystemExit, lambda: N.merge_lora(str(tinyqwen), str(tpt), str(tmp / "m_near"), rounding="nearest", min_delta_kept=0.95), "below")
    check("refused merge leaves no output directory", not (tmp / "m_near").exists())
    rec2 = N.merge_lora(str(tinyqwen), str(tpt), str(tmp / "m_sr"), rounding="stochastic", seed=1, min_delta_kept=0.5)
    check("stochastic rounding keeps the tiny delta", rec2["delta_kept"] > 0.8, f"{rec2['delta_kept']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unpacked sglang 0.5.20 package directory")
    ap.add_argument("--hf-patch", default=str(HERE.parent.parent / "negeig_retrofit" / "patch.py"))
    ap.add_argument("--tinyqwen", default=None, help="tiny qwen3_5 checkpoint dir for merge-lora (skipped if absent)")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    src = Path(a.src)
    tmp = Path(tempfile.mkdtemp(prefix="negeig_cpu_test_"))
    try:
        build_stub_packages(src)
        steps = [
            ("pure", lambda: test_pure(Path(a.hf_patch))),
            ("sources", lambda: test_sources(src)),
            ("kernels", lambda: test_kernels(src, tmp)),
            ("wiring", lambda: test_wiring(src, tmp)),
            ("plugin", lambda: test_plugin(tmp)),
            ("export", lambda: test_export(tmp)),
        ]
        if a.tinyqwen:
            steps.append(("merge", lambda: test_merge(Path(a.tinyqwen), tmp)))
        for name, fn in steps:
            if a.only and a.only != name:
                continue
            try:
                fn()
            except BaseException:
                check(f"step {name} ran to the end", False, traceback.format_exc().strip().splitlines()[-1])
                traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [r for r in RESULTS if not r["ok"]]
    print(json.dumps({"passed": len(RESULTS) - len(bad), "failed": len(bad), "failures": [r["name"] for r in bad]}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
