from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from dikwp_continuum2.config import load_identity_hypotheses, load_profile, load_strategies, load_technology_milestones
from dikwp_continuum2.gates import evaluate_strategy_gate
from dikwp_continuum2.identity import IdentityLedger, IdentityNode
from dikwp_continuum2.semantic_mesh import build_continuity_demo_mesh
from dikwp_continuum2.simulator import run_simulation, simulate_trial


class ContinuumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile(ROOT / "examples" / "duan_yucong_synthetic_profile.json")
        cls.strategies = load_strategies(ROOT / "config" / "strategies.json")
        cls.hypotheses = load_identity_hypotheses(ROOT / "config" / "identity_hypotheses.json")
        cls.milestones = load_technology_milestones(ROOT / "config" / "technology_milestones.json")

    def test_profile_has_all_axes(self):
        self.assertEqual(len(self.profile.baseline_axes), 12)
        self.assertIn("现象意识结论保持不确定性标记", self.profile.purpose_contract.protected_invariants)

    def test_mesh_is_networked_and_complete(self):
        audit = build_continuity_demo_mesh().network_compatibility()
        self.assertEqual(audit["transform_coverage"], 1.0)
        self.assertTrue(audit["has_cycle"])
        self.assertEqual(audit["verdict"], "NETWORK_COMPATIBLE")

    def test_destructive_upload_gate_blocks_same_person_claim(self):
        strategy = next(item for item in self.strategies if item.strategy_id == "destructive_upload")
        decision = evaluate_strategy_gate(strategy, self.profile.risk_tolerance)
        self.assertEqual(decision.action, "BLOCK_SAME_PERSON_CLAIM")
        self.assertEqual(decision.identity_claim, "SUCCESSOR_OR_ARCHIVE_ONLY")

    def test_portfolio_gate_is_not_autonomous(self):
        strategy = next(item for item in self.strategies if item.strategy_id == "continuum_portfolio")
        decision = evaluate_strategy_gate(strategy, self.profile.risk_tolerance)
        self.assertIn(decision.action, {"ALLOW_WITH_CLINICAL_REVIEW", "REVIEW"})
        self.assertIn("独立临床与研究伦理复核", decision.required_reviews)

    def test_identity_ledger_rejects_unknown_parent(self):
        ledger = IdentityLedger()
        with self.assertRaises(ValueError):
            ledger.add(IdentityNode("child", ["missing"], 2030, "COPY", {}, "BRANCH", "UNKNOWN"))

    def test_short_simulation_is_deterministic(self):
        strategy = next(item for item in self.strategies if item.strategy_id == "continuum_portfolio")
        one = simulate_trial(self.profile, strategy, self.hypotheses, self.milestones, seed=123)
        two = simulate_trial(self.profile, strategy, self.hypotheses, self.milestones, seed=123)
        self.assertEqual(one.to_dict(), two.to_dict())

    def test_simulation_returns_all_strategies(self):
        summaries, examples = run_simulation(self.profile, self.strategies, self.hypotheses, self.milestones, 4, 99)
        self.assertEqual(len(summaries), len(self.strategies))
        self.assertEqual(set(examples), {item.strategy_id for item in self.strategies})

    def test_upload_cannot_score_robust_continuity_when_executed(self):
        # Search a deterministic seed range for at least one upload event.
        strategy = next(item for item in self.strategies if item.strategy_id == "destructive_upload")
        found = None
        for seed in range(1000, 1400):
            outcome = simulate_trial(self.profile, strategy, self.hypotheses, self.milestones, seed)
            if any("破坏性扫描" in event for event in outcome.events):
                found = outcome
                break
        self.assertIsNotNone(found)
        self.assertEqual(found.final_axes["causal_continuity"], 0.0)
        self.assertEqual(found.robust_score, 0.0)
        self.assertTrue(found.false_immortality)


if __name__ == "__main__":
    unittest.main()
