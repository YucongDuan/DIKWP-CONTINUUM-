from __future__ import annotations

if __package__:
    from ._ui_presentation import localize_html as _ui_localize_html
else:
    from _ui_presentation import localize_html as _ui_localize_html


import csv
import hashlib
import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from .identity import IdentityLedger, IdentityNode
from .models import AXES, GateDecision, Profile, Strategy, StrategySummary, TrialOutcome
from .semantic_mesh import SemanticMesh

AXIS_LABELS = {
    "biological_viability": "生物可生存性",
    "neural_integrity": "神经完整性",
    "causal_continuity": "因果连续性",
    "autobiographical_memory": "自传记忆",
    "personality_values": "人格与价值",
    "agency_purpose": "能动性与目的",
    "embodiment_interoception": "具身与内感受",
    "relational_continuity": "关系连续性",
    "legal_identity": "法律身份",
    "digital_resilience": "数字韧性",
    "recoverability": "可恢复性",
    "phenomenal_evidence": "现象体验证据",
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ui_localize_html(json.dumps(payload, ensure_ascii=False, indent=2)), encoding="utf-8")


def write_strategy_csv(path: Path, summaries: Sequence[StrategySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "strategy_id", "strategy_name", "trials", "mean_robust_score", "median_robust_score",
        "p10_robust_score", "p90_robust_score", "probability_robust_continuity",
        "probability_no_gap_continuity", "probability_archive_survives",
        "probability_false_immortality", "mean_branch_count", "decision_score",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = summary.to_dict()
            writer.writerow({field: row[field] for field in fields})


def mesh_payload(mesh: SemanticMesh) -> Dict[str, object]:
    return {
        "nodes": [asdict(node) for node in mesh.nodes.values()],
        "transforms": [asdict(edge) for edge in mesh.transforms],
        "type_pair_counts": mesh.type_pair_counts(),
        "network_audit": mesh.network_compatibility(),
    }


def build_identity_demo(profile: Profile, examples: Mapping[str, TrialOutcome]) -> IdentityLedger:
    ledger = IdentityLedger()
    ledger.add(
        IdentityNode(
            node_id="DYC-BIO-2026",
            parent_ids=[],
            year=2026,
            operation="BASELINE_BIOLOGICAL_PROCESS",
            continuity_axes=dict(profile.baseline_axes),
            claim="ORIGINAL_BIOLOGICAL_PERSON",
            consent_status="SIGNED_PURPOSE_CONTRACT",
        )
    )
    ledger.add(
        IdentityNode(
            node_id="DYC-EXOCORTEX-2029",
            parent_ids=["DYC-BIO-2026"],
            year=2029,
            operation="REVERSIBLE_MEMORY_MIRROR",
            continuity_axes={
                "autobiographical_memory": 0.70,
                "personality_values": 0.45,
                "agency_purpose": 0.40,
                "digital_resilience": 0.75,
                "causal_continuity": 0.25,
            },
            claim="NON_PERSONAL_COGNITIVE_AID",
            consent_status="REVOCABLE",
        )
    )
    portfolio = examples.get("continuum_portfolio")
    if portfolio:
        ledger.add(
            IdentityNode(
                node_id="DYC-HYBRID-CANDIDATE-2100",
                parent_ids=["DYC-BIO-2026", "DYC-EXOCORTEX-2029"],
                year=2100,
                operation="SYNTHETIC_PORTFOLIO_OUTCOME",
                continuity_axes=portfolio.final_axes,
                claim="CONTINUITY_CANDIDATE_NOT_PROVEN",
                consent_status="ADVISORY_SIMULATION_ONLY",
            )
        )
    upload = examples.get("destructive_upload")
    if upload:
        ledger.add(
            IdentityNode(
                node_id="DYC-UPLOAD-BRANCH-2100",
                parent_ids=["DYC-BIO-2026"],
                year=2100,
                operation="DESTRUCTIVE_SCAN_BRANCH",
                continuity_axes=upload.final_axes,
                claim="SUCCESSOR_OR_ARCHIVE_ONLY",
                consent_status="BLOCKED_AS_SAME_PERSON_CLAIM",
            )
        )
    return ledger


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _bar(value: float, cls: str = "") -> str:
    pct = max(0.0, min(100.0, value * 100.0))
    return f'<div class="bar"><span class="{cls}" style="width:{pct:.2f}%"></span></div>'


def build_dashboard(
    output_path: Path,
    profile: Profile,
    strategies: Sequence[Strategy],
    summaries: Sequence[StrategySummary],
    examples: Mapping[str, TrialOutcome],
    gates: Mapping[str, Mapping[str, object]],
    mesh: SemanticMesh,
    forecasts: Sequence[Mapping[str, object]],
    run_meta: Mapping[str, object],
) -> None:
    strategy_map = {strategy.strategy_id: strategy for strategy in strategies}
    best = summaries[0]
    portfolio = next((s for s in summaries if s.strategy_id == "continuum_portfolio"), best)
    destructive = next((s for s in summaries if s.strategy_id == "destructive_upload"), None)
    mesh_audit = mesh.network_compatibility()

    rows = []
    for rank, summary in enumerate(summaries, start=1):
        gate = gates[summary.strategy_id]
        gate_class = "ok" if str(gate["action"]).startswith("ALLOW") else "warn" if gate["action"] in {"REVIEW", "HOLD"} else "bad"
        rows.append(
            "<tr>"
            f"<td>{rank}</td><td><strong>{html.escape(summary.strategy_name)}</strong><br><code>{summary.strategy_id}</code></td>"
            f"<td>{summary.decision_score:.3f}{_bar(summary.decision_score)}</td>"
            f"<td>{summary.mean_robust_score:.3f}</td>"
            f"<td>{_pct(summary.probability_no_gap_continuity)}</td>"
            f"<td>{_pct(summary.probability_archive_survives)}</td>"
            f"<td>{_pct(summary.probability_false_immortality)}</td>"
            f"<td><span class=\"pill {gate_class}\">{html.escape(str(gate['action']))}</span></td>"
            "</tr>"
        )

    identity_rows = []
    for key, value in portfolio.identity_scores_mean.items():
        identity_rows.append(
            f"<tr><td>{html.escape(key)}</td><td>{value:.3f}</td><td>{_bar(value, 'accent')}</td></tr>"
        )

    axis_rows = []
    portfolio_example = examples.get("continuum_portfolio")
    if portfolio_example:
        for axis in AXES:
            start = profile.baseline_axes.get(axis, 0.0)
            end = portfolio_example.final_axes.get(axis, 0.0)
            axis_rows.append(
                f"<tr><td>{AXIS_LABELS[axis]}</td><td>{start:.3f}</td><td>{end:.3f}</td>"
                f"<td>{_bar(end, 'accent2')}</td></tr>"
            )

    forecast_cards = []
    for forecast in forecasts:
        forecast_cards.append(
            '<article class="forecast">'
            f"<div class=\"forecast-top\"><span>{html.escape(str(forecast['forecast_id']))}</span>"
            f"<strong>{float(forecast['probability'])*100:.0f}%</strong></div>"
            f"<h4>{html.escape(str(forecast['event']))}</h4>"
            f"<p>结算窗口：{html.escape(str(forecast['resolution_year']))} · 状态：种子先验</p>"
            "</article>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DIKWP-CONTINUUM² Dashboard</title>
<style>
:root{{--ink:#152033;--muted:#667085;--paper:#f5f7fb;--card:#fff;--line:#dde3ec;--navy:#173b63;--teal:#087e8b;--gold:#bd7b17;--red:#a33a3a;--green:#237a57}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,"Noto Sans SC","Microsoft YaHei",sans-serif;background:var(--paper);color:var(--ink);line-height:1.55}}
header{{background:linear-gradient(120deg,#102d4e,#184d68 65%,#087e8b);color:#fff;padding:44px 7vw 36px}}
header h1{{margin:0;font-size:clamp(30px,4vw,54px);letter-spacing:.02em}}header p{{max-width:980px;font-size:18px;color:#e9f2f5}}
main{{max-width:1440px;margin:auto;padding:28px 4vw 70px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 7px 25px rgba(20,40,70,.06)}}.metric b{{display:block;font-size:32px;color:var(--navy)}}.metric span{{color:var(--muted)}}
section{{margin-top:26px}}h2{{font-size:25px;margin:0 0 14px}}h3{{margin-top:0}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:12px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#eef3f8;position:sticky;top:0}}code{{font-size:12px;color:#667085}}
.bar{{height:8px;background:#edf0f5;border-radius:99px;overflow:hidden;margin-top:6px;min-width:110px}}.bar span{{display:block;height:100%;background:var(--teal)}}.bar span.accent{{background:var(--gold)}}.bar span.accent2{{background:var(--navy)}}
.pill{{display:inline-block;padding:4px 8px;border-radius:99px;font-size:11px;font-weight:700}}.pill.ok{{background:#ddf4e9;color:#176340}}.pill.warn{{background:#fff1d3;color:#865807}}.pill.bad{{background:#fde2e2;color:#8c2424}}
.callout{{border-left:6px solid var(--gold);background:#fff9eb;padding:18px 22px;border-radius:12px}}.twocol{{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}}.forecasts{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.forecast{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px}}.forecast-top{{display:flex;justify-content:space-between;color:var(--muted)}}.forecast-top strong{{font-size:25px;color:var(--teal)}}.forecast h4{{margin:8px 0}}.forecast p{{color:var(--muted);margin:0;font-size:13px}}
.small{{color:var(--muted);font-size:13px}}footer{{margin-top:30px;color:var(--muted);font-size:12px}}@media(max-width:1000px){{.grid{{grid-template-columns:repeat(2,1fr)}}.twocol{{grid-template-columns:1fr}}.forecasts{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:620px){{.grid,.forecasts{{grid-template-columns:1fr}}header{{padding:32px 20px}}main{{padding:20px 14px}}table{{font-size:12px}}}}
</style></head><body>
<header><h1>DIKWP-CONTINUUM²</h1><p>跨基质生命连续性、身份保真与开放式长生系统。目标不是让一个副本长期运行，而是在不确定的身份理论与未来技术条件下，最大化生命过程、神经因果链、目的、关系、责任与可恢复性的联合连续。</p></header>
<main>
<div class="callout"><strong>系统决策：</strong>以生物—神经连续为主轴，以可撤销外脑和数字孪生为扩展，以器官修复和神经假体为桥，以渐进、重叠、可回退的跨基质迁移为条件性前沿。破坏性扫描只能生成后继或档案，不能被标记为已证明的本人永生。</div>
<section class="grid">
<div class="card metric"><span>模拟策略</span><b>{len(summaries)}</b><small>覆盖生物、外脑、假体、混合、上传与停滞</small></div>
<div class="card metric"><span>组合路线无断裂连续率</span><b>{_pct(portfolio.probability_no_gap_continuity)}</b><small>合成情景，不是个人寿命预测</small></div>
<div class="card metric"><span>组合路线档案存续率</span><b>{_pct(portfolio.probability_archive_survives)}</b><small>档案存续不等于本人存续</small></div>
<div class="card metric"><span>DIKWP×DIKWP覆盖</span><b>{mesh_audit['transform_coverage']*100:.0f}%</b><small>{mesh_audit['verdict']} · {mesh_audit['observer_count']}类观察者</small></div>
</section>
<section><h2>八条路线的联合评估</h2><div class="card" style="overflow:auto"><table><thead><tr><th>#</th><th>路线</th><th>综合决策分</th><th>稳健连续分</th><th>无断裂</th><th>档案存续</th><th>伪永生风险</th><th>动作门</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="twocol"><div class="card"><h2>组合路线：六种身份理论</h2><table><thead><tr><th>身份假说</th><th>平均分</th><th>可视化</th></tr></thead><tbody>{''.join(identity_rows)}</tbody></table></div>
<div class="card"><h2>解释规则</h2><p><strong>稳健连续分</strong>取六种身份理论的最低值，因此功能模式高保真不能补偿生物或神经因果链完全中断。</p><p><strong>伪永生</strong>表示数字档案或分支继续存在，但原始生命闭环已不可逆终止。</p><p><strong>现象体验证据</strong>只是第三人称证据轴，系统永久保留第一人称连续性的不可消除残差。</p><p class="small">运行种子：{run_meta['seed']} · 每策略试验：{run_meta['trials_per_strategy']} · 时间范围：{profile.horizon_start}–{profile.horizon_end}</p></div></section>
<section><h2>组合路线示范轨迹的十二轴终态</h2><div class="card" style="overflow:auto"><table><thead><tr><th>连续性轴</th><th>2026基线</th><th>2100示范终态</th><th>终态</th></tr></thead><tbody>{''.join(axis_rows)}</tbody></table></div></section>
<section><h2>预测合同：需要未来公开结算的种子先验</h2><div class="forecasts">{''.join(forecast_cards)}</div></section>
<section class="twocol"><div class="card"><h2>系统必须守住的边界</h2><ul><li>不根据未知健康信息给段玉聪做医学诊断或处方。</li><li>不把未经批准的抗衰老、干细胞、外泌体或基因干预当作可实施方案。</li><li>不允许数字分支自行批准其本人身份、财产控制或永久复制。</li><li>任何不可逆神经操作均需独立临床、伦理、身份和法律复核。</li><li>所有记忆写入、删除、合并、来源与授权进入可审计谱系。</li></ul></div>
<div class="card"><h2>Purpose Contract摘要</h2><p>{html.escape(profile.purpose_contract.declared_goal)}</p><p><strong>版本：</strong>{html.escape(profile.purpose_contract.version)}　<strong>风险容忍：</strong>{profile.risk_tolerance:.2f}</p><p class="small">本档案为合成示范，不含真实健康、年龄、基因或医疗数据。</p></div></section>
<footer>DIKWP-CONTINUUM² offline dashboard · generated {html.escape(str(run_meta['generated_at']))} · research and governance prototype only</footer>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_ui_localize_html(html_text), encoding="utf-8")


def write_manifest(root: Path, output_file: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output_file:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    output_file.write_text(_ui_localize_html("\n".join(lines) + "\n"), encoding="utf-8")
