from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class UncertaintySpec:
    kind: str  # 'uniform', 'normal', 'lognormal'
    params: Dict[str, Any]  # e.g., {"lower": -0.1, "upper": 0.1} for multiplicative ±10%


@dataclass
class VariableSpec:
    name: str
    sources: List[str]  # tags of input rasters feeding this variable
    aggregation: Dict[str, Any]  # method + options
    uncertainty: Optional[UncertaintySpec] = None
    ratios: Optional[Dict[str, Any]] = None  # optional components split spec
    writer: Dict[str, Any] = field(default_factory=lambda: {"type": "chm"})


@dataclass
class ExperimentSpec:
    name: str
    N: int
    seed: int
    variables: List[VariableSpec]
    inputs: Dict[str, Any]
    workspace: Dict[str, Any]
    run: Dict[str, Any]
    provenance: Dict[str, Any] = field(default_factory=dict)


def from_dict(d: Dict[str, Any]) -> ExperimentSpec:
    variables = [
        VariableSpec(
            name=v["name"],
            sources=v.get("sources", []),
            aggregation=v.get("aggregation", {}),
            uncertainty=UncertaintySpec(**v["uncertainty"]) if v.get("uncertainty") else None,
            ratios=v.get("ratios"),
            writer=v.get("writer", {"type": "chm"}),
        )
        for v in d.get("variables", [])
    ]
    return ExperimentSpec(
        name=d["name"],
        N=int(d["N"]),
        seed=int(d.get("seed", 0)),
        variables=variables,
        inputs=d.get("inputs", {}),
        workspace=d.get("workspace", {}),
        run=d.get("run", {}),
        provenance=d.get("provenance", {}),
    )

