"""Conditional full-budget FFN implemented as a sparse-Cayley program tree."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitSparseSkewBasis(nn.Module):
    """Degree-bounded skew operator with formula-generated perfect matchings."""

    # Each triple forms one connected 3-regular graph.  The reversible
    # quadratic-affine-quadratic permutations have exact diameters
    # (D=384: 10,10,10; D=4096: 16,15,15).  Constants are shared executable
    # topology, not checkpoint or per-layer metadata.
    _TOPOLOGY_SEEDS = (21, 124, 233, 93, 160, 191, 18, 31, 38)
    _AFFINE_MULTIPLIERS = (61, 359, 169, 373, 131, 115, 113, 19, 29)

    def __init__(
        self,
        width: int,
        *,
        degree: int,
        alpha: float,
        topology_offset: int,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 2 or degree <= 0:
            raise ValueError("width must be positive/even and degree positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        seeds = self._TOPOLOGY_SEEDS[
            topology_offset : topology_offset + degree
        ]
        multipliers = self._AFFINE_MULTIPLIERS[
            topology_offset : topology_offset + degree
        ]
        if len(seeds) != degree or len(multipliers) != degree:
            raise ValueError("not enough implicit topology formulas")
        if any(math.gcd(multiplier, width) != 1 for multiplier in multipliers):
            raise ValueError("implicit permutation multiplier must be coprime")
        self.width = width
        self.degree = degree
        self.alpha = alpha
        self.topology_seeds = tuple(int(value) for value in seeds)
        self.affine_multipliers = tuple(int(value) for value in multipliers)
        self.raw_edge_weights = nn.Parameter(torch.empty(degree, width // 2))
        nn.init.normal_(self.raw_edge_weights, mean=0.0, std=1.0)

    def bounded_edge_weights(self) -> torch.Tensor:
        return (self.alpha / self.degree) * torch.tanh(self.raw_edge_weights)

    def _permutation(
        self, positions: torch.Tensor, matching: int
    ) -> torch.Tensor:
        seed = self.topology_seeds[matching]
        multiplier = self.affine_multipliers[matching]
        left_quadratic = 6 * (2 * seed + 1)
        left_offset = 97 * seed + 17
        right_quadratic = 6 * (2 * ((53 * seed + 7) % 257) + 1)
        right_offset = 193 * seed + 29
        permuted = torch.remainder(
            positions + left_quadratic * positions.square(), self.width
        )
        permuted = torch.remainder(
            multiplier * permuted + left_offset, self.width
        )
        return torch.remainder(
            permuted + right_quadratic * permuted.square() + right_offset,
            self.width,
        )

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[-1] != self.width:
            raise ValueError("values must have shape [tokens, width]")
        positions = torch.arange(
            self.width, dtype=torch.long, device=values.device
        )
        result = torch.zeros_like(values)
        weights = self.bounded_edge_weights()
        for matching in range(self.degree):
            permutation = self._permutation(positions, matching)
            ordered = values[:, permutation].unflatten(
                -1, (self.width // 2, 2)
            )
            weight = weights[matching]
            updates = torch.stack(
                (weight * ordered[..., 1], -weight * ordered[..., 0]),
                dim=-1,
            ).flatten(-2)
            result = result.index_add(1, permutation, updates)
        return result


class SharedCayleyBases(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        bases: int = 3,
        degree: int = 3,
        alpha: float = 0.25,
        neumann_order: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.width = width
        self.neumann_order = neumann_order
        self.banks = nn.ModuleList(
            [
                ImplicitSparseSkewBasis(
                    width,
                    degree=degree,
                    alpha=alpha,
                    topology_offset=index * degree,
                )
                for index in range(bases)
            ]
        )
        del seed  # Topology is formula-derived; RNG state initializes weights.

    def apply(
        self, values: torch.Tensor, basis_index: int, *, transpose: bool = False
    ) -> torch.Tensor:
        bank = self.banks[basis_index % len(self.banks)]
        source = values
        term = values
        inverse_action = values
        sign = 1.0 if transpose else -1.0
        for _ in range(self.neumann_order):
            term = sign * bank.apply(term)
            inverse_action = inverse_action + term
        return 2.0 * inverse_action - source


class CayleyProgramTreeMLP(nn.Module):
    """A depth-nine binary program tree matching a 384x1024 SwiGLU budget."""

    # Coprime to the frozen width 384 and to power-of-two production widths.
    # The common LCG multiplier 1_103_515_245 is divisible by three and would
    # restrict a width-384 depth to one residue class, causing node collisions.
    _ROUTE_MULTIPLIER = 1_103_515_247
    _ROUTE_DEPTH_STRIDE = 12_345

    def __init__(
        self,
        width: int = 384,
        *,
        depth: int = 9,
        bases: int = 3,
        degree: int = 3,
        alpha: float = 0.25,
        neumann_order: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 2 or depth <= 0:
            raise ValueError("width must be positive/even and depth positive")
        self.width = width
        self.depth = depth
        self.nodes = 2**depth - 1
        self.bases = SharedCayleyBases(
            width,
            bases=bases,
            degree=degree,
            alpha=alpha,
            neumann_order=neumann_order,
            seed=seed,
        )
        self.payload = nn.Parameter(torch.empty(self.nodes, 2, 3, width))
        self.threshold = nn.Parameter(torch.zeros(self.nodes))
        with torch.no_grad():
            nn.init.normal_(self.payload[:, :, 0], mean=1.0, std=0.02)
            nn.init.normal_(self.payload[:, :, 1], mean=1.0, std=0.02)
            nn.init.normal_(
                self.payload[:, :, 2], mean=0.0, std=0.15
            )
        # Route coordinates are an implicit integer hash of the node.  Storing
        # an index per node would violate the exact resident-byte ledger.
        self.route_seed = int(seed + 7919)
        self.force_bits: tuple[int, ...] | None = None
        self.last_nodes: list[torch.Tensor] = []
        self.last_bits: list[torch.Tensor] = []
        self.last_probabilities: list[torch.Tensor] = []
        self.last_router_auxiliary = torch.tensor(0.0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _route(
        self, projected: torch.Tensor, node: torch.Tensor, depth_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coordinates = self._route_coordinates(node, depth_index)
        logits = projected.gather(1, coordinates[:, None]).squeeze(1)
        logits = logits - self.threshold[node]
        probabilities = torch.sigmoid(logits.float()).to(projected.dtype)
        if self.force_bits is not None:
            bit_value = self.force_bits[depth_index] % 2
            hard = torch.full_like(probabilities, float(bit_value))
            selector = hard
        else:
            hard = (probabilities >= 0.5).to(projected.dtype)
            selector = hard + probabilities - probabilities.detach()
        return selector, hard.to(torch.long), probabilities

    def _route_coordinates(
        self, node: torch.Tensor, depth_index: int
    ) -> torch.Tensor:
        return torch.remainder(
            node * self._ROUTE_MULTIPLIER
            + self.route_seed
            + depth_index * self._ROUTE_DEPTH_STRIDE,
            self.width,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        initial = hidden_states.reshape(-1, self.width)
        hidden = initial
        node = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        node_rows = []
        bit_rows = []
        probability_rows = []
        auxiliaries = []
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
            delta = self.bases.apply(local_delta, depth_index, transpose=True)
            hidden = hidden + scale * delta
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
            node = 2 * node + 1 + hard_bit
        self.last_nodes = node_rows
        self.last_bits = bit_rows
        self.last_probabilities = probability_rows
        self.last_router_auxiliary = torch.stack(auxiliaries).mean()
        return (hidden - initial).reshape(original_shape)


def exact_ledger(
    width: int = 384,
    intermediate_width: int = 1024,
    depth: int = 9,
    bases: int = 3,
    degree: int = 3,
    neumann_order: int = 4,
) -> dict[str, int | float]:
    nodes = 2**depth - 1
    payload = nodes * 2 * 3 * width
    generators = bases * degree * width // 2
    thresholds = nodes
    candidate = payload + generators + thresholds
    baseline = 3 * width * intermediate_width
    active = depth * (2 * neumann_order * degree * width + 4 * width)
    return {
        "nodes": nodes,
        "payload_parameters": payload,
        "generator_parameters": generators,
        "threshold_parameters": thresholds,
        "candidate_parameters": candidate,
        "baseline_swiglu_parameters": baseline,
        "parameter_difference": candidate - baseline,
        "hard_path_payload_scalars": depth * 3 * width,
        "candidate_multiply_like_active_upper_bound": active,
        "baseline_dense_macs": baseline,
        "ideal_active_ratio": active / baseline,
    }
