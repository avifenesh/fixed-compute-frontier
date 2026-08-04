from __future__ import annotations

import unittest

import torch

from experiments.triangular_value_encoding_autograd_hybrid import (
    triangular_value_encoding_autograd_hybrid,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TriangularValueEncodingAutogradHybridTest(unittest.TestCase):
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

        actual = triangular_value_encoding_autograd_hybrid(
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


if __name__ == "__main__":
    unittest.main()
