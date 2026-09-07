from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import IdentityHypothesis, Profile, PurposeContract, Strategy


def _read_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_identity_hypotheses(path: str | Path) -> List[IdentityHypothesis]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("identity hypotheses must be a list")
    return [IdentityHypothesis(**item) for item in data]


def load_strategies(path: str | Path) -> List[Strategy]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("strategies must be a list")
    return [Strategy(**item) for item in data]


def load_profile(path: str | Path) -> Profile:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("profile must be an object")
    contract_raw = data.pop("purpose_contract")
    contract = PurposeContract(**contract_raw)
    return Profile(purpose_contract=contract, **data)


def load_technology_milestones(path: str | Path) -> Dict[str, Dict[str, float]]:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("technology milestones must be an object")
    return data
