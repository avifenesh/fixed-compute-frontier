from __future__ import annotations

import unittest

import torch

from experiments.shared_projection_dictionary_gate import (
    byte_ledger,
    fit_line_dictionary,
    random_control,
    reconstruction_error,
)


class SharedProjectionDictionaryGateTests(unittest.TestCase):
    def test_exact_repeated_lines_need_only_the_source_count(self) -> None:
        generator = torch.Generator().manual_seed(1)
        atoms = torch.randn((4, 12), generator=generator)
        atoms = atoms / atoms.norm(dim=1, keepdim=True)
        assignments = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        scales = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.7, 0.2])
        rows = scales[:, None] * atoms[assignments]
        fit = fit_line_dictionary(rows, 4, seed=7, iterations=8)
        self.assertLess(fit["relative_frobenius_error"], 1e-5)

    def test_reconstruction_is_sign_invariant(self) -> None:
        atoms = torch.eye(3)
        rows = torch.cat((atoms, -2 * atoms), dim=0)
        error, assignments, coefficients = reconstruction_error(rows, atoms)
        self.assertLess(error, 1e-7)
        self.assertEqual(assignments.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(coefficients.tolist(), [1, 1, 1, -2, -2, -2])

    def test_random_control_preserves_row_norms(self) -> None:
        rows = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)
        control = random_control(rows, seed=9)
        self.assertTrue(torch.allclose(rows.norm(dim=1), control.norm(dim=1)))
        self.assertFalse(torch.allclose(rows, control))

    def test_ledger_counts_dictionary_gather_scheme(self) -> None:
        ledger = byte_ledger(row_count=700, width=100, atom_count=100)
        self.assertEqual(ledger["original_bf16_bits"], 1_120_000)
        self.assertEqual(ledger["index_bits_per_row"], 7)
        self.assertLess(ledger["encoded_to_original_ratio"], 0.16)
        self.assertLess(ledger["ideal_macs_ratio"], 0.16)

    def test_invalid_atom_count_is_rejected(self) -> None:
        rows = torch.eye(3)
        for count in (0, 4):
            with self.assertRaises(ValueError):
                fit_line_dictionary(rows, count, seed=1)


if __name__ == "__main__":
    unittest.main()
