#!/usr/bin/env python3
"""G1: does fla's chunked gated delta rule stay accurate when beta leaves (0, 1)?

Compares fla.chunk_gated_delta_rule (bf16 inputs, the path transformers uses for
Qwen3.5) against a float64 sequential reference, forward output and input
gradients, for beta drawn from (0, 1) and from (0, 2), with decays near 1.
Pass rule (frozen): for every shape, the (0, 2) relative error is at most twice
the (0, 1) relative error plus 1e-3, forward and backward.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def reference(q, k, v, g, beta):
    """Sequential gated delta rule in float64. q, k: [B,T,H,K]; v: [B,T,H,V]; g, beta: [B,T,H]."""
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    q = F.normalize(q, dim=-1) * q.shape[-1] ** -0.5
    k = F.normalize(k, dim=-1)
    B, T, H, K = k.shape
    V = v.shape[-1]
    state = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    out = torch.empty(B, T, H, V, dtype=torch.float64, device=q.device)
    for t in range(T):
        state = state * g[:, t].exp()[..., None, None]
        kv = (state * k[:, t, :, :, None]).sum(-2)
        delta = (v[:, t] - kv) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        out[:, t] = (state * q[:, t, :, :, None]).sum(-2)
    return out


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def run(T, H, K, V, beta_hi, seed, alpha_lo):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(1, T, H, K, device=dev)
    k = torch.randn(1, T, H, K, device=dev)
    v = torch.randn(1, T, H, V, device=dev)
    # log decay: alpha = exp(g) uniform in (alpha_lo, 1)
    g = torch.empty(1, T, H, device=dev).uniform_(alpha_lo, 1.0).log()
    beta = torch.rand(1, T, H, device=dev) * beta_hi
    if beta_hi > 1:
        # force a quarter of the steps close to a full reflection
        mask = torch.rand(1, T, H, device=dev) < 0.25
        beta = torch.where(mask, 1.95 + 0.05 * torch.rand_like(beta), beta)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, g, beta)]
    ql, kl, vl, gl, bl = leaves
    o, _ = chunk_gated_delta_rule(
        ql.bfloat16(), kl.bfloat16(), vl.bfloat16(), gl.float(), bl.bfloat16(),
        use_qk_l2norm_in_kernel=True, output_final_state=False,
    )
    do = torch.randn_like(o, dtype=torch.float32)
    (o.float() * do).sum().backward()
    grads = [x.grad.detach().clone() for x in leaves]
    ref_leaves = [x.detach().clone().double().requires_grad_(True) for x in (q, k, v, g, beta)]
    ro = reference(*ref_leaves)
    (ro * do.double()).sum().backward()
    names = ["q", "k", "v", "g", "beta"]
    res = {"fwd": rel(o, ro)}
    for n, gr, rl in zip(names, grads, ref_leaves):
        res[f"d{n}"] = rel(gr, rl.grad)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    ok = True
    for T in (256, 1024):
        for alpha_lo in (0.9, 0.99):
            for seed in (0, 1):
                base = run(T, 4, 128, 128, 1.0, seed, alpha_lo)
                wide = run(T, 4, 128, 128, 2.0, seed, alpha_lo)
                for key in base:
                    passed = wide[key] <= 2 * base[key] + 1e-3
                    ok &= passed
                    rows.append({"T": T, "alpha_lo": alpha_lo, "seed": seed, "metric": key,
                                 "rel_err_beta01": base[key], "rel_err_beta02": wide[key], "pass": passed})
                print(T, alpha_lo, seed, {k: (round(base[k], 5), round(wide[k], 5)) for k in base}, flush=True)
    json.dump({"pass": ok, "rows": rows}, open(args.out, "w"), indent=1)
    print("G1 PASS" if ok else "G1 FAIL")


if __name__ == "__main__":
    main()
