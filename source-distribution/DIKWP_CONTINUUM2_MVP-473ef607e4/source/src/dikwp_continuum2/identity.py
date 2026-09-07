from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from .models import AXES, ContinuityState, IdentityHypothesis


@dataclass(frozen=True)
class IdentityNode:
    node_id: str
    parent_ids: List[str]
    year: int
    operation: str
    continuity_axes: Dict[str, float]
    claim: str
    consent_status: str


class IdentityLedger:
    """A lineage DAG for body, neural, hybrid and digital continuations."""

    def __init__(self) -> None:
        self.nodes: Dict[str, IdentityNode] = {}

    def add(self, node: IdentityNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate identity node {node.node_id}")
        for parent in node.parent_ids:
            if parent not in self.nodes:
                raise ValueError(f"unknown parent {parent}")
        self.nodes[node.node_id] = node
        if self._has_cycle():
            del self.nodes[node.node_id]
            raise ValueError("identity ledger must remain acyclic")

    def _has_cycle(self) -> bool:
        children: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        for node_id, node in self.nodes.items():
            for parent in node.parent_ids:
                children.setdefault(parent, []).append(node_id)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            for child in children.get(node_id, []):
                if visit(child):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return any(visit(node_id) for node_id in self.nodes if node_id not in visited)

    def to_dict(self) -> Dict[str, object]:
        return {"nodes": [asdict(node) for node in self.nodes.values()]}


def classify_identity_claim(state: ContinuityState, hypothesis_scores: Dict[str, float]) -> str:
    if state.irreversible_break and state.branch_count > 0:
        return "COPY_SURVIVES_ORIGINAL_BREAK"
    if state.suspended and not state.alive:
        return "SUSPENDED_RECOVERABILITY_CANDIDATE"
    if min(
        state.axes.get("neural_integrity", 0.0),
        state.axes.get("causal_continuity", 0.0),
        state.axes.get("autobiographical_memory", 0.0),
        state.axes.get("agency_purpose", 0.0),
    ) >= 0.70:
        return "STRONG_CONTINUITY_CANDIDATE"
    if sum(hypothesis_scores.values()) / max(1, len(hypothesis_scores)) >= 0.60:
        return "MULTI_THEORY_SUCCESSOR"
    return "ARCHIVE_OR_DERIVATIVE_ONLY"


def hypothesis_scores(state: ContinuityState, hypotheses: Iterable[IdentityHypothesis]) -> Dict[str, float]:
    return {hypothesis.hypothesis_id: hypothesis.score(state.axes) for hypothesis in hypotheses}
