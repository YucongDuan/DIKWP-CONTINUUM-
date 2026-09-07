from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

from .config import load_identity_hypotheses, load_profile, load_strategies, load_technology_milestones
from .forecasts import seed_forecast_contracts
from .gates import gate_catalog
from .reporting import (
    build_dashboard,
    build_identity_demo,
    mesh_payload,
    write_json,
    write_manifest,
    write_strategy_csv,
)
from .semantic_mesh import build_continuity_demo_mesh
from .simulator import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DIKWP-CONTINUUM² continuity simulator")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--trials", type=int, default=1500, help="trials per strategy")
    parser.add_argument("--seed", type=int, default=7152026)
    parser.add_argument("--profile", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    profile_path = args.profile or root / "examples" / "duan_yucong_synthetic_profile.json"
    profile = load_profile(profile_path)
    strategies = load_strategies(root / "config" / "strategies.json")
    hypotheses = load_identity_hypotheses(root / "config" / "identity_hypotheses.json")
    milestones = load_technology_milestones(root / "config" / "technology_milestones.json")
    summaries, examples = run_simulation(
        profile=profile,
        strategies=strategies,
        hypotheses=hypotheses,
        milestones=milestones,
        trials_per_strategy=args.trials,
        seed=args.seed,
    )
    gates = gate_catalog(strategies, profile.risk_tolerance)
    mesh = build_continuity_demo_mesh()
    forecasts = seed_forecast_contracts()
    ledger = build_identity_demo(profile, examples)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    output = root / "outputs"
    output.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "system": "DIKWP-CONTINUUM²",
        "version": "1.0.0",
        "generated_at": generated_at,
        "seed": args.seed,
        "trials_per_strategy": args.trials,
        "total_trials": args.trials * len(strategies),
        "profile_id": profile.profile_id,
        "warning": "Synthetic research simulation; not a medical prognosis or treatment recommendation.",
    }
    write_json(output / "strategy_summaries.json", [item.to_dict() for item in summaries])
    write_strategy_csv(output / "strategy_summaries.csv", summaries)
    write_json(output / "example_trials.json", {key: value.to_dict() for key, value in examples.items()})
    write_json(output / "gate_catalog.json", gates)
    write_json(output / "semantic_mesh.json", mesh_payload(mesh))
    write_json(output / "identity_ledger.json", ledger.to_dict())
    write_json(output / "forecast_contracts.json", forecasts)
    write_json(output / "RUN_PROOF.json", run_meta)
    build_dashboard(
        output / "dashboard.html", profile, strategies, summaries, examples, gates, mesh, forecasts, run_meta
    )
    write_manifest(root, root / "MANIFEST.sha256")

    print(json.dumps({
        "output_dir": str(output),
        "top_strategy": summaries[0].to_dict(),
        "mesh_audit": mesh.network_compatibility(),
        "run_meta": run_meta,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
