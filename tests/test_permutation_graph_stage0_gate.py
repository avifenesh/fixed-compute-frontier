from __future__ import annotations

import unittest

from experiments.permutation_graph_stage0_gate import (
    _assert_independent_sources,
    crosscheck_world,
)


class PermutationGraphStage0GateTests(unittest.TestCase):
    def test_sources_are_independent(self) -> None:
        hashes = _assert_independent_sources()
        self.assertEqual(len(set(hashes.values())), len(hashes))
        self.assertTrue(all(len(digest) == 64 for digest in hashes.values()))

    def test_independent_interpreters_agree_on_trace_and_queries(self) -> None:
        result = crosscheck_world(seed=17, command_count=128)
        self.assertEqual(result["commands"], 128)
        self.assertEqual(result["arena_bytes"], 4096)
        self.assertEqual(result["query_checks"], 4480)
        self.assertEqual(result["directed_semantic_checks"]["checks"], 10)
        self.assertEqual(result["directed_semantic_checks"]["accepted_two_switches"], 2)
        self.assertEqual(
            result["accepted_commands"] + result["rejected_commands"], 128
        )


if __name__ == "__main__":
    unittest.main()
