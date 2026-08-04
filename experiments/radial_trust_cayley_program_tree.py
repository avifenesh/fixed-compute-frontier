"""Norm-bounded, full-rank successor to the Cayley program tree v1."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.cayley_program_tree import CayleyProgramTreeMLP


class RadialTrustCayleyProgramTreeMLP(CayleyProgramTreeMLP):
    """Cayley program tree with a basis-invariant trust region per edge.

    For a raw update ``d``, the map

        S(d) = d / sqrt(1 + mean(d**2) / trust_rms**2)

    has RMS strictly below ``trust_rms``.  Its Jacobian is full rank for every
    finite ``d``; unlike clipping, it introduces no zero-derivative boundary.
    """

    def __init__(self, *args: object, trust_rms: float = 1.0, **kwargs: object) -> None:
        if trust_rms <= 0.0:
            raise ValueError("trust_rms must be positive")
        super().__init__(*args, **kwargs)
        self.trust_rms = float(trust_rms)
        self.record_diagnostics = False
        self.last_trust_diagnostics: list[dict[str, float]] = []

    def radial_trust_region(self, delta: torch.Tensor) -> torch.Tensor:
        squared_rms = delta.float().square().mean(dim=-1, keepdim=True)
        denominator = torch.sqrt(1.0 + squared_rms / self.trust_rms**2)
        return delta / denominator.to(delta.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        initial = hidden_states.reshape(-1, self.width)
        hidden = initial
        node = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        node_rows = []
        bit_rows = []
        probability_rows = []
        auxiliaries = []
        diagnostics = []
        scale = 1.0 / math.sqrt(self.depth)
        for depth_index in range(self.depth):
            projected = self.bases.apply(hidden, depth_index)
            selector, hard_bit, probabilities = self._route(
                projected, node, depth_index
            )
            if self.training and self.force_bits is None:
                choices = self.payload[node]
                selected = (
                    choices[:, 0] * (1.0 - selector[:, None, None])
                    + choices[:, 1] * selector[:, None, None]
                )
            else:
                selected = self.payload[node, hard_bit]
            gate, up, down = selected.unbind(dim=1)
            activation = F.silu(gate * projected) * (up * projected)
            local_delta = down * activation
            raw_delta = self.bases.apply(
                local_delta, depth_index, transpose=True
            )
            trusted_delta = self.radial_trust_region(raw_delta)
            hidden = hidden + scale * trusted_delta
            node_rows.append(node.detach())
            bit_rows.append(hard_bit.detach())
            probability_rows.append(probabilities.detach())
            mean_probability = probabilities.float().mean()
            auxiliaries.append(
                2.0
                * (
                    mean_probability.square()
                    + (1.0 - mean_probability).square()
                )
            )
            if self.record_diagnostics:
                with torch.no_grad():
                    diagnostics.append(
                        {
                            "raw_delta_rms": float(
                                raw_delta.float().square().mean(dim=-1).sqrt().max()
                            ),
                            "trusted_delta_rms": float(
                                trusted_delta.float()
                                .square()
                                .mean(dim=-1)
                                .sqrt()
                                .max()
                            ),
                            "hidden_rms": float(
                                hidden.float().square().mean(dim=-1).sqrt().max()
                            ),
                            "hidden_abs_max": float(hidden.float().abs().max()),
                        }
                    )
            node = 2 * node + 1 + hard_bit
        self.last_nodes = node_rows
        self.last_bits = bit_rows
        self.last_probabilities = probability_rows
        self.last_router_auxiliary = torch.stack(auxiliaries).mean()
        self.last_trust_diagnostics = diagnostics
        return (hidden - initial).reshape(original_shape)
