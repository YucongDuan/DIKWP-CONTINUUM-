from __future__ import annotations

from typing import Iterable, List

from .models import GateDecision, Strategy


GRADE_ORDER = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "X": 0}


def evaluate_strategy_gate(strategy: Strategy, risk_tolerance: float) -> GateDecision:
    reasons: List[str] = []
    reviews: List[str] = []

    grade = GRADE_ORDER.get(strategy.evidence_grade.upper(), 0)
    if grade >= 4 and strategy.reversibility >= 0.75 and not strategy.destructive:
        action = "ALLOW_WITH_CLINICAL_REVIEW"
    elif strategy.destructive:
        action = "BLOCK_SAME_PERSON_CLAIM"
    elif grade <= 1:
        action = "RESEARCH_ONLY"
    elif strategy.reversibility < 0.45:
        action = "HOLD"
    else:
        action = "REVIEW"

    if strategy.destructive:
        reasons.append("该方案会破坏原有生物或神经载体，无法把复制成功等同于本人继续存在。")
    if strategy.creates_branch:
        reasons.append("该方案产生分叉后继，必须建立独立身份、责任与退出权。")
    if strategy.reversibility < 0.6:
        reasons.append("可逆性不足，若失败可能造成不可恢复的因果连续中断。")
    if grade <= 2:
        reasons.append("证据仍以动物、早期人体或理论研究为主，不能作为常规个人实施方案。")
    if risk_tolerance < 0.35 and (strategy.destructive or strategy.reversibility < 0.7):
        reasons.append("个人风险容忍度低于该方案所需阈值。")

    if strategy.requires_human_review:
        reviews.append("独立临床与研究伦理复核")
    if strategy.creates_branch or strategy.destructive:
        reviews.extend(["身份连续委员会", "法律受托人与家属代表"])
    if "neural" in strategy.strategy_id or "hybrid" in strategy.strategy_id or "upload" in strategy.strategy_id:
        reviews.append("神经技术与意识证据委员会")

    if strategy.destructive:
        identity_claim = "SUCCESSOR_OR_ARCHIVE_ONLY"
    elif strategy.creates_branch:
        identity_claim = "BRANCHED_SUCCESSOR"
    elif strategy.strategy_id in {"gradual_hybrid", "continuum_portfolio"}:
        identity_claim = "CONTINUITY_CANDIDATE_NOT_PROVEN"
    else:
        identity_claim = "BIOLOGICAL_OR_FUNCTIONAL_CONTINUITY"

    if not reasons:
        reasons.append("当前证据与可逆性允许在合格专业人员监督下作为连续性增益措施。")

    return GateDecision(
        action=action,
        reasons=reasons,
        required_reviews=sorted(set(reviews)),
        identity_claim=identity_claim,
    )


def gate_catalog(strategies: Iterable[Strategy], risk_tolerance: float) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for strategy in strategies:
        decision = evaluate_strategy_gate(strategy, risk_tolerance)
        result[strategy.strategy_id] = {
            "action": decision.action,
            "reasons": decision.reasons,
            "required_reviews": decision.required_reviews,
            "identity_claim": decision.identity_claim,
        }
    return result
