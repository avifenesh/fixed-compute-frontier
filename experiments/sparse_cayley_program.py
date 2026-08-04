"""Trainable token-conditioned programs over sparse Cayley expert banks."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseSkewExpertBank(nn.Module):
    """Sparse skew generators on deterministic perfect-match topologies."""

    def __init__(
        self,
        width: int,
        experts: int,
        degree: int,
        alpha: float,
        seed: int,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 2 or experts <= 0 or degree <= 0:
            raise ValueError("width must be positive/even; experts and degree positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.width = width
        self.experts = experts
        self.degree = degree
        self.alpha = alpha
        rng = np.random.default_rng(seed)
        permutations = np.empty((experts, degree, width), dtype=np.int64)
        inverses = np.empty_like(permutations)
        for expert in range(experts):
            for matching in range(degree):
                permutation = rng.permutation(width)
                permutations[expert, matching] = permutation
                inverses[expert, matching, permutation] = np.arange(width)
        self.register_buffer("permutations", torch.from_numpy(permutations), persistent=True)
        self.register_buffer("inverses", torch.from_numpy(inverses), persistent=True)
        self.raw_edge_weights = nn.Parameter(
            torch.empty(experts, degree, width // 2)
        )
        nn.init.normal_(self.raw_edge_weights, mean=0.0, std=0.25)

    def bounded_edge_weights(self) -> torch.Tensor:
        return (self.alpha / self.degree) * torch.tanh(self.raw_edge_weights)

    def apply_all(self, values: torch.Tensor) -> torch.Tensor:
        """Return every expert's A_e x as [tokens, experts, width]."""
        if values.ndim != 2 or values.shape[-1] != self.width:
            raise ValueError("values must have shape [tokens, width]")
        tokens = values.shape[0]
        expanded = values[:, None, None, :].expand(
            tokens, self.experts, self.degree, self.width
        )
        permutation = self.permutations[None].expand(tokens, -1, -1, -1)
        ordered = torch.gather(expanded, -1, permutation)
        pairs = ordered.unflatten(-1, (self.width // 2, 2))
        weights = self.bounded_edge_weights()[None]
        updates = torch.stack(
            (weights * pairs[..., 1], -weights * pairs[..., 0]), dim=-1
        ).flatten(-2)
        inverse = self.inverses[None].expand(tokens, -1, -1, -1)
        canonical = torch.gather(updates, -1, inverse)
        return canonical.sum(dim=2)

    def apply_mixture(
        self, values: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        if expert_weights.shape != (values.shape[0], self.experts):
            raise ValueError("expert weights must have shape [tokens, experts]")
        return torch.einsum("ned,ne->nd", self.apply_all(values), expert_weights)


class CayleyProgramProjection(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        experts: int = 4,
        degree: int = 3,
        route_length: int = 2,
        neumann_order: int = 4,
        alpha: float = 0.25,
        seed: int = 0,
        static_route: bool = False,
    ) -> None:
        super().__init__()
        self.width = width
        self.experts = experts
        self.route_length = route_length
        self.neumann_order = neumann_order
        self.static_route = static_route
        self.bank = SparseSkewExpertBank(width, experts, degree, alpha, seed)
        self.router = nn.Parameter(torch.empty(route_length, experts, width))
        nn.init.normal_(self.router, mean=0.0, std=1.0 / math.sqrt(width))
        self.input_scale = nn.Parameter(torch.ones(width))
        self.output_scale = nn.Parameter(torch.ones(width))
        self.last_route_probabilities: list[torch.Tensor] = []
        self.last_route_ids: list[torch.Tensor] = []
        self.last_router_auxiliary = torch.tensor(0.0)
        self.force_route: int | None = None

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _route(self, values: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(values, self.router[step])
        probabilities = torch.softmax(logits.float(), dim=-1).to(values.dtype)
        if self.force_route is not None:
            ids = torch.full(
                (values.shape[0],), self.force_route % self.experts,
                dtype=torch.long, device=values.device,
            )
            weights = F.one_hot(ids, self.experts).to(values.dtype)
        elif self.static_route:
            ids = torch.full(
                (values.shape[0],), step % self.experts,
                dtype=torch.long, device=values.device,
            )
            weights = F.one_hot(ids, self.experts).to(values.dtype)
            # Keep the stored router in the graph without changing the forward.
            weights = weights + probabilities - probabilities.detach()
        else:
            ids = probabilities.argmax(dim=-1)
            hard = F.one_hot(ids, self.experts).to(values.dtype)
            weights = hard + probabilities - probabilities.detach()
        return weights, probabilities

    def _cayley_apply(
        self, values: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        source = values
        term = values
        inverse_action = values
        for _ in range(self.neumann_order):
            term = -self.bank.apply_mixture(term, expert_weights)
            inverse_action = inverse_action + term
        return 2.0 * inverse_action - source

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        values = (hidden_states * self.input_scale).reshape(-1, self.width)
        probabilities_record = []
        ids_record = []
        auxiliaries = []
        for step in range(self.route_length):
            weights, probabilities = self._route(values, step)
            values = self._cayley_apply(values, weights)
            probabilities_record.append(probabilities.detach())
            ids_record.append(weights.detach().argmax(dim=-1))
            importance = probabilities.float().mean(dim=0)
            auxiliaries.append(self.experts * importance.square().sum())
        self.last_route_probabilities = probabilities_record
        self.last_route_ids = ids_record
        self.last_router_auxiliary = torch.stack(auxiliaries).mean()
        return values.reshape(shape) * self.output_scale


class SparseCayleyProgramMLP(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        branches: int = 2,
        experts: int = 4,
        degree: int = 3,
        route_length: int = 2,
        neumann_order: int = 4,
        alpha: float = 0.25,
        seed: int = 0,
        static_route: bool = False,
        baseline_intermediate_width: int = 1024,
    ) -> None:
        super().__init__()
        self.width = width
        self.branches = branches
        self.branch_scale = math.sqrt(
            baseline_intermediate_width / (branches * width)
        )
        projections = []
        for branch in range(branches):
            row = nn.ModuleDict()
            for index, name in enumerate(("gate", "up", "down")):
                row[name] = CayleyProgramProjection(
                    width,
                    experts=experts,
                    degree=degree,
                    route_length=route_length,
                    neumann_order=neumann_order,
                    alpha=alpha,
                    seed=seed + 101 * branch + 17 * index,
                    static_route=static_route,
                )
            projections.append(row)
        self.projections = nn.ModuleList(projections)

    def program_projections(self) -> list[CayleyProgramProjection]:
        return [module for row in self.projections for module in row.values()]

    def router_auxiliary_loss(self) -> torch.Tensor:
        return torch.stack(
            [module.last_router_auxiliary for module in self.program_projections()]
        ).mean()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = []
        for row in self.projections:
            gate = row["gate"](hidden_states)
            up = row["up"](hidden_states)
            outputs.append(row["down"](F.silu(gate) * up))
        return self.branch_scale * torch.stack(outputs).sum(dim=0)


def logical_ledger(
    width: int = 384,
    experts: int = 4,
    degree: int = 3,
    route_length: int = 2,
    neumann_order: int = 4,
    branches: int = 2,
    baseline_intermediate_width: int = 1024,
) -> dict[str, int | float]:
    projection_parameters = (
        route_length * experts * width
        + experts * degree * width // 2
        + 2 * width
    )
    projections = 3 * branches
    candidate_parameters = projections * projection_parameters
    narrow_width = candidate_parameters // (3 * width)
    selected_edge_multiplies = route_length * neumann_order * degree * width
    router_macs = route_length * experts * width
    diagonal_multiplies = 2 * width
    candidate_active = projections * (
        selected_edge_multiplies + router_macs + diagonal_multiplies
    )
    baseline_dense = 3 * width * baseline_intermediate_width
    return {
        "projection_parameters": projection_parameters,
        "candidate_ffn_parameters": candidate_parameters,
        "equal_dense_swiglu_width": narrow_width,
        "equal_dense_swiglu_parameters": 3 * width * narrow_width,
        "candidate_ideal_active_scalar_ops": candidate_active,
        "baseline_dense_macs": baseline_dense,
        "candidate_to_baseline_ideal_ratio": candidate_active / baseline_dense,
    }
