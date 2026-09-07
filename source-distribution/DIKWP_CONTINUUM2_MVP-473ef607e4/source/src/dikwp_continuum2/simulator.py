from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .identity import hypothesis_scores
from .models import AXES, ContinuityState, IdentityHypothesis, Profile, Strategy, StrategySummary, TrialOutcome, clamp


EVIDENCE_FACTOR = {"A": 1.0, "B": 0.88, "C": 0.72, "D": 0.52, "E": 0.30, "X": 0.12}


def logistic(year: int, midpoint: float, slope: float) -> float:
    return 1.0 / (1.0 + math.exp(-slope * (year - midpoint)))


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _annual_baseline_drift(state: ContinuityState, year_index: int, rng: random.Random) -> None:
    age_pressure = 1.0 + year_index / 70.0
    state.update_axis("biological_viability", -(0.0060 * age_pressure + rng.uniform(-0.001, 0.0015)))
    state.update_axis("neural_integrity", -(0.0046 * age_pressure + rng.uniform(-0.0008, 0.0012)))
    state.update_axis("embodiment_interoception", -(0.0022 * age_pressure + rng.uniform(-0.0005, 0.0008)))
    state.update_axis("recoverability", -(0.0012 * age_pressure))
    state.update_axis("autobiographical_memory", -max(0.0, 0.0012 * age_pressure + rng.uniform(-0.0004, 0.0006)))
    state.update_axis("relational_continuity", rng.uniform(-0.0012, 0.0010))
    state.update_axis("agency_purpose", rng.uniform(-0.0008, 0.0009))
    state.update_axis("personality_values", rng.uniform(-0.0005, 0.0005))
    state.update_axis("digital_resilience", -0.0003)
    state.update_axis("legal_identity", -0.0001)
    # The phenomenal evidence axis is evidence about continuity, not a direct measure of experience.
    state.update_axis("phenomenal_evidence", -0.0010 * age_pressure)


def _apply_strategy_effects(state: ContinuityState, strategy: Strategy, rng: random.Random) -> None:
    if state.year < strategy.start_year or state.suspended:
        return
    maturity = logistic(state.year, strategy.maturity_midpoint, strategy.maturity_slope)
    evidence_factor = EVIDENCE_FACTOR.get(strategy.evidence_grade.upper(), 0.4)
    effect_scale = 0.42 * maturity * evidence_factor * rng.uniform(0.72, 1.18)

    # Portfolio effects are deliberately capped: combining options does not add their benefits linearly.
    if strategy.strategy_id == "continuum_portfolio":
        effect_scale *= 0.92

    for axis, annual_delta in strategy.annual_effects.items():
        state.update_axis(axis, annual_delta * effect_scale)

    if strategy.strategy_id == "exocortex" and state.year == 2030 and state.branch_count == 0:
        state.branch_count += 1
        state.audit_events.append("2030: 建立非人格化外脑分支；仅作为记忆与决策证据镜像。")
    if strategy.strategy_id == "continuum_portfolio" and state.year == 2029 and state.branch_count == 0:
        state.branch_count += 1
        state.audit_events.append("2029: 建立可撤销的个人外脑分支，未授予本人身份。")

    if strategy.strategy_id == "gradual_hybrid" and state.year >= 2050:
        if rng.random() < 0.025 * maturity:
            state.branch_count += 1
            state.update_axis("causal_continuity", 0.05)
            state.update_axis("phenomenal_evidence", 0.03)
            state.audit_events.append(f"{state.year}: 完成一次重叠式神经—数字功能迁移实验。")

    if strategy.strategy_id == "continuum_portfolio" and state.year >= 2052:
        if rng.random() < 0.015 * maturity:
            state.branch_count += 1
            state.update_axis("causal_continuity", 0.035)
            state.update_axis("digital_resilience", 0.04)
            state.audit_events.append(f"{state.year}: 通过双向重叠测试扩展混合认知节点。")


def _apply_shocks(state: ContinuityState, strategy: Strategy, year_index: int, rng: random.Random) -> None:
    if state.suspended:
        return
    age_pressure = 1.0 + year_index / 55.0
    bio_prob = min(0.18, 0.017 * age_pressure)
    neural_prob = min(0.12, 0.010 * age_pressure)
    data_prob = 0.020
    relationship_prob = 0.012

    if rng.random() < bio_prob:
        raw = rng.uniform(0.06, 0.24) * age_pressure
        protection = strategy.shock_protection.get("biological", 0.0)
        loss = raw * (1.0 - protection)
        state.update_axis("biological_viability", -loss)
        state.update_axis("recoverability", -0.25 * loss)
        state.audit_events.append(f"{state.year}: 生物系统冲击，净损失 {loss:.3f}。")

    if rng.random() < neural_prob:
        raw = rng.uniform(0.05, 0.20) * age_pressure
        protection = strategy.shock_protection.get("neural", 0.0)
        loss = raw * (1.0 - protection)
        state.update_axis("neural_integrity", -loss)
        state.update_axis("autobiographical_memory", -0.50 * loss)
        state.update_axis("causal_continuity", -0.20 * loss)
        state.update_axis("phenomenal_evidence", -0.35 * loss)
        state.audit_events.append(f"{state.year}: 神经系统冲击，净损失 {loss:.3f}。")

    if rng.random() < data_prob:
        raw = rng.uniform(0.05, 0.18)
        protection = strategy.shock_protection.get("data", 0.0)
        loss = raw * (1.0 - protection)
        state.update_axis("digital_resilience", -loss)
        state.update_axis("autobiographical_memory", -0.20 * loss)
        state.update_axis("legal_identity", -0.10 * loss)
        state.audit_events.append(f"{state.year}: 数据/密钥故障，净损失 {loss:.3f}。")

    if rng.random() < relationship_prob:
        loss = rng.uniform(0.02, 0.08)
        state.update_axis("relational_continuity", -loss)
        state.update_axis("agency_purpose", -0.15 * loss)
        state.audit_events.append(f"{state.year}: 关系网络中断，净损失 {loss:.3f}。")


def _apply_technology_milestones(
    state: ContinuityState,
    strategy: Strategy,
    milestones: Mapping[str, Mapping[str, float]],
    rng: random.Random,
    achieved: set[str],
) -> None:
    if state.suspended:
        return
    for name, spec in milestones.items():
        if name in achieved:
            continue
        probability = logistic(state.year, spec["midpoint"], spec["slope"]) * 0.10
        if rng.random() >= probability:
            continue
        impact = spec["impact"]
        achieved.add(name)
        if name == "health_digital_twin":
            state.update_axis("biological_viability", impact * 0.35)
            state.update_axis("recoverability", impact * 0.25)
        elif name == "organ_banking_and_replacement":
            state.update_axis("biological_viability", impact * 0.60)
            state.update_axis("recoverability", impact * 0.50)
        elif name == "long_term_home_bci":
            state.update_axis("agency_purpose", impact * 0.45)
            state.update_axis("relational_continuity", impact * 0.35)
            state.update_axis("neural_integrity", impact * 0.20)
        elif name == "cognitive_neuroprosthesis":
            state.update_axis("autobiographical_memory", impact * 0.50)
            state.update_axis("causal_continuity", impact * 0.35)
        elif name == "non_destructive_brain_state_mapping":
            state.update_axis("digital_resilience", impact * 0.55)
            state.update_axis("phenomenal_evidence", impact * 0.30)
        elif name == "gradual_substrate_replacement":
            if strategy.strategy_id in {"gradual_hybrid", "continuum_portfolio"}:
                state.update_axis("causal_continuity", impact * 0.60)
                state.update_axis("recoverability", impact * 0.45)
        elif name == "whole_brain_emulation":
            state.update_axis("digital_resilience", impact * 0.65)
            if strategy.strategy_id == "destructive_upload":
                state.update_axis("autobiographical_memory", impact * 0.50)
        elif name == "legal_branch_personhood":
            state.update_axis("legal_identity", impact * 0.70)
        state.audit_events.append(f"{state.year}: 技术里程碑出现：{name}。")


def _handle_destructive_upload(state: ContinuityState, strategy: Strategy, rng: random.Random) -> None:
    if strategy.strategy_id != "destructive_upload" or state.irreversible_break or state.year < 2050:
        return
    maturity = logistic(state.year, strategy.maturity_midpoint, strategy.maturity_slope)
    if rng.random() < 0.025 * maturity:
        fidelity = clamp(rng.gauss(0.82 * maturity + 0.12, 0.08))
        state.branch_count += 1
        state.axes["autobiographical_memory"] = max(state.axes["autobiographical_memory"], fidelity)
        state.axes["personality_values"] = max(state.axes["personality_values"], fidelity * 0.96)
        state.axes["agency_purpose"] = max(state.axes["agency_purpose"], fidelity * 0.92)
        state.axes["digital_resilience"] = max(state.axes["digital_resilience"], 0.70 + 0.20 * maturity)
        state.axes["relational_continuity"] = max(state.axes["relational_continuity"], fidelity * 0.70)
        state.axes["legal_identity"] = max(state.axes["legal_identity"], 0.50 + 0.25 * maturity)
        state.axes["biological_viability"] = 0.0
        state.axes["neural_integrity"] = 0.0
        state.axes["causal_continuity"] = 0.0
        state.axes["embodiment_interoception"] *= 0.15
        state.axes["phenomenal_evidence"] *= 0.20
        state.alive = False
        state.irreversible_break = True
        state.audit_events.append(
            f"{state.year}: 执行破坏性扫描并生成数字分支；功能保真度 {fidelity:.3f}，原神经因果链终止。"
        )


def _handle_biostasis(state: ContinuityState, strategy: Strategy, rng: random.Random) -> None:
    if strategy.strategy_id not in {"biostasis", "continuum_portfolio"} or state.suspended:
        return
    crisis = state.axes["biological_viability"] < 0.12 or state.axes["neural_integrity"] < 0.10
    if not crisis:
        return
    maturity = logistic(state.year, 2050 if strategy.strategy_id == "biostasis" else 2045, 0.10)
    capture_probability = 0.25 + 0.45 * maturity
    if rng.random() < capture_probability:
        state.suspended = True
        state.alive = False
        state.axes["recoverability"] = max(state.axes["recoverability"], 0.25 + 0.35 * maturity)
        state.axes["causal_continuity"] = max(state.axes["causal_continuity"], 0.20 + 0.25 * maturity)
        state.axes["neural_integrity"] = max(state.axes["neural_integrity"], 0.18 + 0.25 * maturity)
        state.audit_events.append(f"{state.year}: 进入生物停滞状态；仅保留未来可恢复候选。")
    else:
        state.alive = False
        state.irreversible_break = True
        state.audit_events.append(f"{state.year}: 危机超出停滞系统捕获能力。")


def _attempt_revival(state: ContinuityState, strategy: Strategy, rng: random.Random) -> None:
    if not state.suspended or state.year < 2065:
        return
    maturity = logistic(state.year, 2082, 0.09)
    if rng.random() < 0.010 * maturity * state.axes["recoverability"]:
        state.suspended = False
        state.alive = True
        state.axes["biological_viability"] = max(state.axes["biological_viability"], 0.45 * maturity)
        state.axes["neural_integrity"] = max(state.axes["neural_integrity"], 0.42 * maturity)
        state.axes["causal_continuity"] = max(state.axes["causal_continuity"], 0.36 * maturity)
        state.axes["phenomenal_evidence"] = max(state.axes["phenomenal_evidence"], 0.28 * maturity)
        state.audit_events.append(f"{state.year}: 合成情景中完成复苏；身份与体验仍需重新评估。")


def simulate_trial(
    profile: Profile,
    strategy: Strategy,
    hypotheses: Iterable[IdentityHypothesis],
    milestones: Mapping[str, Mapping[str, float]],
    seed: int,
) -> TrialOutcome:
    rng = random.Random(seed)
    state = ContinuityState(year=profile.horizon_start, axes=dict(profile.baseline_axes))
    achieved: set[str] = set()

    for year in range(profile.horizon_start, profile.horizon_end + 1):
        state.year = year
        if state.irreversible_break and not state.suspended:
            # Digital descendants can persist, but the original continuity process no longer evolves.
            state.update_axis("digital_resilience", 0.003 * rng.uniform(0.4, 1.2))
            state.update_axis("legal_identity", rng.uniform(-0.001, 0.002))
            continue

        if state.suspended:
            state.update_axis("recoverability", -0.0015)
            state.update_axis("digital_resilience", -0.0005)
            _attempt_revival(state, strategy, rng)
            continue

        year_index = year - profile.horizon_start
        _annual_baseline_drift(state, year_index, rng)
        _apply_strategy_effects(state, strategy, rng)
        _apply_shocks(state, strategy, year_index, rng)
        _apply_technology_milestones(state, strategy, milestones, rng, achieved)
        _handle_destructive_upload(state, strategy, rng)
        _handle_biostasis(state, strategy, rng)

        if state.axes["biological_viability"] <= 0.03 or state.axes["neural_integrity"] <= 0.02:
            if strategy.strategy_id in {"biostasis", "continuum_portfolio"}:
                _handle_biostasis(state, strategy, rng)
            if not state.suspended:
                state.alive = False
                state.irreversible_break = True
                state.audit_events.append(f"{state.year}: 原生生命闭环终止。")

    scores = hypothesis_scores(state, hypotheses)
    robust = min(scores.values()) if scores else 0.0
    archive_survives = state.axes["digital_resilience"] >= 0.60 and state.axes["autobiographical_memory"] >= 0.50
    false_immortality = archive_survives and state.irreversible_break and state.branch_count > 0
    return TrialOutcome(
        strategy_id=strategy.strategy_id,
        final_year=profile.horizon_end,
        final_axes={axis: round(state.axes.get(axis, 0.0), 6) for axis in AXES},
        hypothesis_scores={key: round(value, 6) for key, value in scores.items()},
        robust_score=round(robust, 6),
        alive=state.alive,
        suspended=state.suspended,
        branch_count=state.branch_count,
        irreversible_break=state.irreversible_break,
        false_immortality=false_immortality,
        events=state.audit_events[-12:],
    )


def summarize_strategy(strategy: Strategy, outcomes: Sequence[TrialOutcome]) -> StrategySummary:
    robust_values = [outcome.robust_score for outcome in outcomes]
    identity_keys = sorted(outcomes[0].hypothesis_scores) if outcomes else []
    identity_means = {
        key: statistics.fmean(outcome.hypothesis_scores[key] for outcome in outcomes) for key in identity_keys
    }
    probability_no_gap = statistics.fmean(
        1.0
        if outcome.alive
        and not outcome.irreversible_break
        and outcome.final_axes.get("causal_continuity", 0.0) >= 0.55
        and outcome.final_axes.get("biological_viability", 0.0) >= 0.35
        and outcome.final_axes.get("neural_integrity", 0.0) >= 0.45
        and outcome.final_axes.get("agency_purpose", 0.0) >= 0.60
        else 0.0
        for outcome in outcomes
    )
    probability_archive = statistics.fmean(
        1.0
        if outcome.final_axes.get("digital_resilience", 0.0) >= 0.60
        and outcome.final_axes.get("autobiographical_memory", 0.0) >= 0.50
        else 0.0
        for outcome in outcomes
    )
    evidence_norm = EVIDENCE_FACTOR.get(strategy.evidence_grade.upper(), 0.0)
    false_rate = statistics.fmean(1.0 if outcome.false_immortality else 0.0 for outcome in outcomes)
    decision_score = (
        0.35 * statistics.fmean(robust_values)
        + 0.20 * probability_no_gap
        + 0.15 * probability_archive
        + 0.10 * (1.0 - false_rate)
        + 0.12 * strategy.reversibility
        + 0.08 * evidence_norm
    )
    return StrategySummary(
        strategy_id=strategy.strategy_id,
        strategy_name=strategy.name,
        trials=len(outcomes),
        mean_robust_score=round(statistics.fmean(robust_values), 6),
        median_robust_score=round(statistics.median(robust_values), 6),
        p10_robust_score=round(percentile(robust_values, 0.10), 6),
        p90_robust_score=round(percentile(robust_values, 0.90), 6),
        probability_robust_continuity=round(statistics.fmean(1.0 if value >= 0.55 else 0.0 for value in robust_values), 6),
        probability_no_gap_continuity=round(probability_no_gap, 6),
        probability_archive_survives=round(probability_archive, 6),
        probability_false_immortality=round(false_rate, 6),
        mean_branch_count=round(statistics.fmean(outcome.branch_count for outcome in outcomes), 6),
        decision_score=round(decision_score, 6),
        identity_scores_mean={key: round(value, 6) for key, value in identity_means.items()},
    )


def run_simulation(
    profile: Profile,
    strategies: Sequence[Strategy],
    hypotheses: Sequence[IdentityHypothesis],
    milestones: Mapping[str, Mapping[str, float]],
    trials_per_strategy: int,
    seed: int,
) -> Tuple[List[StrategySummary], Dict[str, TrialOutcome]]:
    summaries: List[StrategySummary] = []
    examples: Dict[str, TrialOutcome] = {}
    for strategy_index, strategy in enumerate(strategies):
        outcomes = [
            simulate_trial(
                profile,
                strategy,
                hypotheses,
                milestones,
                seed + strategy_index * 100_000 + trial_index,
            )
            for trial_index in range(trials_per_strategy)
        ]
        summaries.append(summarize_strategy(strategy, outcomes))
        examples[strategy.strategy_id] = outcomes[0]
    summaries.sort(key=lambda item: item.decision_score, reverse=True)
    return summaries, examples
