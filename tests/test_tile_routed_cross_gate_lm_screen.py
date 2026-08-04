from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tile_routed_cross_gate_lm_screen import (
    TILE_SIZE,
    CrossGateMLP,
    cyclic_candidates,
    effective_alpha,
    routed_up,
)


class TileRoutedCrossGateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.gate = torch.randn(2, 3, 2 * TILE_SIZE)
        self.up = torch.randn_like(self.gate)

    def test_zero_beta_is_exact_endpoint(self) -> None:
        alpha = effective_alpha(torch.zeros(()))
        for arm in ("static_local", "global_routed", "tile_routed"):
            selected, _ = routed_up(self.gate, self.up, arm)
            endpoint = self.up + alpha * (selected - self.up)
            torch.testing.assert_close(endpoint, self.up, rtol=0.0, atol=0.0)

    def test_bf16_endpoint_preserves_dtype_and_bits(self) -> None:
        gate = self.gate.to(torch.bfloat16)
        up = self.up.to(torch.bfloat16)
        alpha = effective_alpha(torch.zeros((), dtype=torch.float32)).to(up.dtype)
        for arm in ("static_balanced", "global_routed", "tile_routed"):
            selected, _ = routed_up(gate, up, arm)
            endpoint = up + alpha * (selected - up)
            self.assertEqual(endpoint.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(endpoint, up))

    def test_candidates_are_tile_local_cyclic_shifts(self) -> None:
        candidates = cyclic_candidates(self.up)
        tiled = self.up.reshape(2, 3, 2, TILE_SIZE)
        torch.testing.assert_close(candidates[..., 0, :], tiled)
        torch.testing.assert_close(
            candidates[..., 1, :], torch.roll(tiled, shifts=-1, dims=-1)
        )

    def test_static_route_is_shift_one(self) -> None:
        selected, routes = routed_up(self.gate, self.up, "static_local")
        expected = torch.roll(
            self.up.reshape(2, 3, 2, TILE_SIZE), shifts=-1, dims=-1
        ).reshape_as(self.up)
        torch.testing.assert_close(selected, expected)
        self.assertTrue(torch.all(routes == 1))

    def test_balanced_static_routes_vary_by_tile_not_token(self) -> None:
        selected, routes = routed_up(self.gate, self.up, "static_balanced")
        self.assertTrue(torch.all(routes[..., 0] == 0))
        self.assertTrue(torch.all(routes[..., 1] == 1))
        candidates = cyclic_candidates(self.up)
        expected = torch.stack(
            (candidates[..., 0, 0, :], candidates[..., 1, 1, :]), dim=-2
        ).reshape_as(self.up)
        torch.testing.assert_close(selected, expected, rtol=0.0, atol=0.0)

    def test_hard_forward_selects_exact_candidates(self) -> None:
        for arm in ("global_routed", "tile_routed"):
            selected, routes = routed_up(self.gate, self.up, arm)
            candidates = cyclic_candidates(self.up)
            expected = torch.gather(
                candidates,
                -2,
                routes.unsqueeze(-1).unsqueeze(-1).expand(
                    *routes.shape, 1, TILE_SIZE
                ),
            ).squeeze(-2).reshape_as(self.up)
            torch.testing.assert_close(selected, expected, rtol=0.0, atol=0.0)

    def test_straight_through_routes_send_gate_gradient(self) -> None:
        gate = self.gate.clone().requires_grad_(True)
        selected, _ = routed_up(gate, self.up, "tile_routed")
        selected.square().mean().backward()
        self.assertGreater(float(gate.grad.abs().sum()), 0.0)

    def test_logical_gauge_reindex_endpoint_is_bit_exact(self) -> None:
        class TinyMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(5, TILE_SIZE, bias=False)
                self.up_proj = nn.Linear(5, TILE_SIZE, bias=False)
                self.down_proj = nn.Linear(TILE_SIZE, 5, bias=False)
                self.act_fn = F.silu

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return self.down_proj(F.silu(self.gate_proj(values)) * self.up_proj(values))

        original = TinyMLP().to(torch.bfloat16)
        values = torch.randn(7, 5)
        values = values.to(torch.bfloat16)
        expected = original(values)
        model = CrossGateMLP(original, "tile_routed_reindexed", reindex_seed=19)
        actual = model(values)
        self.assertEqual(model.reindex_permutation.unique().numel(), TILE_SIZE)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
