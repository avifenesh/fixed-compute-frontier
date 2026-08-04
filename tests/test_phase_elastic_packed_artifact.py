import torch

from experiments.phase_elastic_packed_artifact import (
    PackedInt4Weight,
    PackedResidualWeight,
    pack_int4,
    pack_ternary,
    unpack_int4,
    unpack_ternary,
)
from experiments.phase_elastic_precision_ffn_lm_screen import QuantizedWeight
from experiments.phase_elastic_residual_precision_v2_lm_screen import ResidualQuantizedWeight


def test_ternary_and_int4_bitpacking_round_trip_exactly():
    ternary = torch.tensor([-1, 0, 1, -1, 1, 0, -1, 1], dtype=torch.int8)
    int4 = torch.tensor([-7, -1, 0, 7, 3, -4], dtype=torch.int8)
    assert torch.equal(unpack_ternary(pack_ternary(ternary), ternary.numel()), ternary)
    assert torch.equal(unpack_int4(pack_int4(int4), int4.numel()), int4)


def test_packed_residual_weight_replays_exact_export_without_parameters():
    torch.manual_seed(7)
    source = ResidualQuantizedWeight(torch.randn(8, 12))
    packed = PackedResidualWeight.from_training(source)
    assert torch.equal(packed(), source.exact())
    assert sum(parameter.numel() for parameter in packed.parameters()) == 0
    assert packed.payload_bytes() == 96 + 24 + 16 + 16


def test_packed_int4_weight_replays_exact_export_without_parameters():
    torch.manual_seed(9)
    source = QuantizedWeight(torch.randn(8, 12), 7)
    packed = PackedInt4Weight.from_training(source)
    assert torch.equal(packed(), source.exact())
    assert sum(parameter.numel() for parameter in packed.parameters()) == 0
    assert packed.payload_bytes() == 48 + 16


def test_packed_int4_audit_exposes_reserved_negative_eight():
    codes = torch.tensor([-8, -7, 0, 7], dtype=torch.int8)
    packed = PackedInt4Weight((1, 4), pack_int4(codes), torch.ones(1))
    assert packed.audit()["code_min"] == -8
