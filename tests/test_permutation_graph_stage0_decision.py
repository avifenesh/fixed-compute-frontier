from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = json.loads(
    (ROOT / "results/permutation-graph-refinement-stage0-decision.json").read_text()
)


class PermutationGraphStage0DecisionTests(unittest.TestCase):
    def test_fatal_control_closes_the_superiority_gate(self) -> None:
        control = DECISION["fatal_control"]
        self.assertEqual(control["whole_episode_trajectory_accuracy"], 1.0)
        self.assertEqual(control["candidate_accuracy_upper_bound"], 1.0)
        self.assertEqual(control["maximum_possible_candidate_advantage_points"], 0.0)
        self.assertGreater(control["required_candidate_advantage_points"], 0.0)
        self.assertFalse(control["gate_reachable"])

    def test_no_gpu_training_or_candidate_claim(self) -> None:
        self.assertEqual(DECISION["status"], "closed_at_stage_0_fatal_control")
        self.assertFalse(DECISION["gpu_rented"])
        self.assertFalse(DECISION["training_run"])
        self.assertFalse(DECISION["decision"]["admit_candidate_004"])

    def test_exact_logical_state_reconciles(self) -> None:
        state = DECISION["fatal_control"]["logical_task_state_bytes"]
        self.assertEqual(
            state["total"],
            state["Z"] + state["two_involutions"] + state["parents"] + state["status"],
        )
        self.assertEqual(state["total"], 514)


if __name__ == "__main__":
    unittest.main()
