from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Mapping, Optional


AXES: tuple[str, ...] = (
    "biological_viability",
    "neural_integrity",
    "causal_continuity",
    "autobiographical_memory",
    "personality_values",
    "agency_purpose",
    "embodiment_interoception",
    "relational_continuity",
    "legal_identity",
    "digital_resilience",
    "recoverability",
    "phenomenal_evidence",
)

DIKWP_TYPES: tuple[str, ...] = ("D", "I", "K", "W", "P")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    claim: str
    source_type: str
    grade: str
    confidence: float
    reversible: bool
    notes: str = ""


@dataclass(frozen=True)
class PurposeContract:
    contract_id: str
    declared_goal: str
    protected_invariants: List[str]
    forbidden_actions: List[str]
    review_triggers: List[str]
    authorized_reviewers: List[str]
    version: str = "1.0"


@dataclass(frozen=True)
class IdentityHypothesis:
    hypothesis_id: str
    name: str
    description: str
    axis_weights: Dict[str, float]
    hard_requirements: Dict[str, float] = field(default_factory=dict)

    def score(self, axes: Mapping[str, float]) -> float:
        if any(axes.get(axis, 0.0) < threshold for axis, threshold in self.hard_requirements.items()):
            return 0.0
        total = sum(self.axis_weights.values()) or 1.0
        return sum(self.axis_weights.get(axis, 0.0) * axes.get(axis, 0.0) for axis in AXES) / total


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    name: str
    description: str
    start_year: int
    maturity_midpoint: int
    maturity_slope: float
    annual_effects: Dict[str, float]
    shock_protection: Dict[str, float]
    evidence_grade: str
    reversibility: float
    destructive: bool = False
    creates_branch: bool = False
    requires_human_review: bool = True
    max_autonomy: str = "RESEARCH_ONLY"


@dataclass
class ContinuityState:
    year: int
    axes: Dict[str, float]
    alive: bool = True
    suspended: bool = False
    branch_count: int = 0
    irreversible_break: bool = False
    audit_events: List[str] = field(default_factory=list)

    def update_axis(self, axis: str, delta: float) -> None:
        self.axes[axis] = clamp(self.axes.get(axis, 0.0) + delta)

    def copy(self) -> "ContinuityState":
        return ContinuityState(
            year=self.year,
            axes=dict(self.axes),
            alive=self.alive,
            suspended=self.suspended,
            branch_count=self.branch_count,
            irreversible_break=self.irreversible_break,
            audit_events=list(self.audit_events),
        )


@dataclass(frozen=True)
class Profile:
    profile_id: str
    display_name: str
    horizon_start: int
    horizon_end: int
    baseline_axes: Dict[str, float]
    purpose_contract: PurposeContract
    risk_tolerance: float = 0.25
    notes: str = ""


@dataclass(frozen=True)
class GateDecision:
    action: str
    reasons: List[str]
    required_reviews: List[str]
    identity_claim: str


@dataclass
class TrialOutcome:
    strategy_id: str
    final_year: int
    final_axes: Dict[str, float]
    hypothesis_scores: Dict[str, float]
    robust_score: float
    alive: bool
    suspended: bool
    branch_count: int
    irreversible_break: bool
    false_immortality: bool
    events: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class StrategySummary:
    strategy_id: str
    strategy_name: str
    trials: int
    mean_robust_score: float
    median_robust_score: float
    p10_robust_score: float
    p90_robust_score: float
    probability_robust_continuity: float
    probability_no_gap_continuity: float
    probability_archive_survives: float
    probability_false_immortality: float
    mean_branch_count: float
    decision_score: float
    identity_scores_mean: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    text: str
    dikwp_mix: Dict[str, float]
    observer: str
    context: str
    provenance: str


@dataclass(frozen=True)
class SemanticTransform:
    transform_id: str
    source_node: str
    target_node: str
    source_type: str
    target_type: str
    observer: str
    context: str
    rationale: str
    confidence: float
