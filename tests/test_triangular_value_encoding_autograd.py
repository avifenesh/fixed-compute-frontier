from __future__ import annotations

import unittest

import torch

from experiments.triangular_value_encoding_attention import serving_value_encoding
from experiments.triangular_value_encoding_autograd import triangular_value_encoding_autograd


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TriangularValueEncodingAutogradTest(unittest.TestCase):
    def test_one_physical_slot_has_exact_local_forward_and_backward_support(self) -> None:
        batch, kv_heads, tokens, head_dim, query_groups = 1, 1, 2, 16, 2
        hidden = kv_heads * query_groups * head_dim
        values = torch.zeros(
            batch, kv_heads, tokens, head_dim,
            device="cuda", dtype=torch.bfloat16, requires_grad=True,
        )
        with torch.no_grad():
            values[0, 0, 0, 2] = 2.0
            values[0, 0, 1, 2] = -3.0
        output_weight = torch.zeros(
            hidden, hidden, device="cuda", dtype=torch.float32, requires_grad=True,
        )
        with torch.no_grad():
            output_weight[5, 2] = 0.125

        actual = triangular_value_encoding_autograd(
            values, output_weight, query_groups=query_groups
        )
        expected = values.detach().clone()
        expected[0, 0, 0, 5] = 4.0
        expected[0, 0, 1, 5] = -9.0
        self.assertTrue(torch.equal(actual, expected))

        upstream = torch.zeros_like(actual)
        upstream[0, 0, 0, 5] = 1.0
        upstream[0, 0, 1, 5] = 2.0
        actual.backward(upstream)
        expected_value_gradient = torch.zeros_like(values)
        expected_value_gradient[0, 0, 0, 2] = 4.0
        expected_value_gradient[0, 0, 0, 5] = 1.0
        expected_value_gradient[0, 0, 1, 2] = 12.0
        expected_value_gradient[0, 0, 1, 5] = 2.0
        self.assertTrue(torch.equal(values.grad, expected_value_gradient))
        self.assertEqual(output_weight.grad.count_nonzero().item(), 1)
        self.assertEqual(output_weight.grad[5, 2].item(), -112.0)

    def test_forward_is_bit_exact_and_gradients_match_surrogate(self) -> None:
        torch.manual_seed(2719)
        batch, kv_heads, tokens, head_dim, query_groups = 2, 2, 65, 64, 3
        hidden = kv_heads * query_groups * head_dim
        values = torch.randn(
            batch, kv_heads, tokens, head_dim,
            device="cuda", dtype=torch.bfloat16, requires_grad=True,
        )
        output_weight = torch.randn(
            hidden, hidden, device="cuda", dtype=torch.float32, requires_grad=True,
        ) * 0.01
        output_weight = output_weight.detach().requires_grad_(True)
        actual = triangular_value_encoding_autograd(
            values, output_weight, query_groups=query_groups
        )
        token_major = values.transpose(1, 2).contiguous().view(
            batch * tokens, kv_heads, head_dim
        )
        expected = serving_value_encoding(
            token_major,
            output_weight.to(torch.bfloat16),
            query_groups=query_groups,
            block_size=16,
            tau=0.125,
        ).view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        self.assertTrue(torch.equal(actual, expected))

        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        actual_value_gradient = values.grad.detach().clone()
        actual_weight_gradient = output_weight.grad.detach().clone()

        reference_values = values.detach().clone().requires_grad_(True)
        reference_weight = output_weight.detach().clone().requires_grad_(True)
        pieces = []
        for start in range(0, head_dim, 16):
            block = reference_values[..., start:start + 16]
            physical = torch.stack([
                reference_weight[
                    start:start + 16,
                    kv_head * query_groups * head_dim + start:
                    kv_head * query_groups * head_dim + start + 16,
                ]
                for kv_head in range(kv_heads)
            ])
            coefficient = (
                torch.tril(physical.to(block.dtype), diagonal=-1).float() / 0.125
            ).to(block.dtype)
            feature = (block.float() * block.float().abs()).to(block.dtype)
            delta = torch.matmul(
                coefficient[None, :, None], feature.unsqueeze(-1)
            ).squeeze(-1)
            pieces.append((block.float() + delta.float()).to(block.dtype))
        torch.cat(pieces, -1).backward(upstream)
        torch.testing.assert_close(
            actual_value_gradient, reference_values.grad, atol=0.03125, rtol=0.0
        )
        torch.testing.assert_close(
            actual_weight_gradient, reference_weight.grad, atol=0.5, rtol=0.01
        )


if __name__ == "__main__":
    unittest.main()
