"""Serializable, evidence-backed teammate-belief snapshots.

This module models beliefs only. Evidence extraction and belief updates are
separate responsibilities, so the model never reads hidden teammate profiles.
"""

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .schema import DEFAULT_SCHEMA, BeliefSchema


class EvidenceSource(str, Enum):
    TEAMMATE_MESSAGE = "teammate_message"
    VISIBLE_OBSERVATION = "visible_observation"
    OWN_ACTION_OUTCOME = "own_action_outcome"
    OWN_MESSAGE = "own_message"
    OWN_ACTION = "own_action"
    VISIBLE_PEER_OUTCOME = "visible_peer_outcome"


def _json_copy(value: Any) -> Any:
    """Reject non-JSON values and detach mutable inputs from stored snapshots."""
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("belief values must be JSON-serializable") from exc


def _probability(value: Optional[float], name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be a number in [0, 1] or None".format(name))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("{} must be in [0, 1]".format(name))


@dataclass(frozen=True)
class Evidence:
    source: EvidenceSource
    step: int
    summary: str
    reference: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, EvidenceSource):
            raise TypeError("source must be an EvidenceSource")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("summary must be a non-empty string")
        if self.reference is not None and not isinstance(self.reference, str):
            raise TypeError("reference must be a string or None")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "step": self.step,
            "summary": self.summary,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls(
            source=EvidenceSource(data["source"]),
            step=data["step"],
            summary=data["summary"],
            reference=data.get("reference"),
        )


@dataclass(frozen=True, init=False)
class DimensionBelief:
    status: str
    _value_json: str = field(repr=False)
    confidence: Optional[float]
    uncertainty: Optional[float]
    evidence: Tuple[Evidence, ...]

    def __init__(
        self,
        status: str = "unknown",
        value: Any = None,
        confidence: Optional[float] = None,
        uncertainty: Optional[float] = None,
        evidence: Sequence[Evidence] = (),
    ) -> None:
        if status not in ("unknown", "supported"):
            raise ValueError("status must be unknown or supported")
        evidence = tuple(evidence)
        if not all(isinstance(item, Evidence) for item in evidence):
            raise TypeError("evidence must contain Evidence records")
        _probability(confidence, "confidence")
        _probability(uncertainty, "uncertainty")
        if status == "unknown":
            if value is not None or confidence is not None:
                raise ValueError("unknown beliefs cannot have a value or confidence")
        elif value is None or confidence is None or not evidence:
            raise ValueError("supported beliefs need a value, confidence, and evidence")
        value_json = json.dumps(_json_copy(value), ensure_ascii=False, sort_keys=True)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "_value_json", value_json)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "evidence", evidence)

    @property
    def value(self) -> Any:
        """Return a detached JSON value so prior snapshots cannot be mutated."""
        return json.loads(self._value_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "value": _json_copy(self.value),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionBelief":
        return cls(
            status=data["status"],
            value=data.get("value"),
            confidence=data.get("confidence"),
            uncertainty=data.get("uncertainty"),
            evidence=tuple(Evidence.from_dict(item) for item in data.get("evidence", [])),
        )


@dataclass(frozen=True)
class BeliefState:
    teammate_id: int
    schema: BeliefSchema = DEFAULT_SCHEMA
    dimensions: Mapping[str, DimensionBelief] = field(default_factory=dict)
    revision: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.teammate_id, bool) or not isinstance(self.teammate_id, int) or self.teammate_id < 1:
            raise ValueError("teammate_id must be a positive integer")
        if not isinstance(self.schema, BeliefSchema):
            raise TypeError("schema must be a BeliefSchema")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        dimensions = dict(self.dimensions)
        if not dimensions:
            dimensions = {name: DimensionBelief() for name in self.schema.names()}
        if set(dimensions) != set(self.schema.names()):
            raise ValueError("belief dimensions must exactly match the schema")
        if not all(isinstance(item, DimensionBelief) for item in dimensions.values()):
            raise TypeError("dimensions must contain DimensionBelief values")
        object.__setattr__(self, "dimensions", MappingProxyType(dimensions))

    def with_evidence(
        self,
        dimension: str,
        value: Any,
        confidence: float,
        evidence: Sequence[Evidence],
        uncertainty: Optional[float] = None,
    ) -> "BeliefState":
        """Return a new snapshot, preserving the prior evidence for this dimension."""
        if dimension not in self.dimensions:
            raise KeyError("unknown dimension: {}".format(dimension))
        new_evidence = tuple(evidence)
        if not new_evidence:
            raise ValueError("an update needs new attributable evidence")
        prior = self.dimensions[dimension]
        updated = DimensionBelief(
            status="supported",
            value=value,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence=prior.evidence + new_evidence,
        )
        dimensions = dict(self.dimensions)
        dimensions[dimension] = updated
        return BeliefState(self.teammate_id, self.schema, dimensions, self.revision + 1)

    def with_unknown(self, dimension: str, evidence: Sequence[Evidence]) -> "BeliefState":
        """Withdraw a prior estimate when new evidence no longer supports it."""
        if dimension not in self.dimensions:
            raise KeyError("unknown dimension: {}".format(dimension))
        new_evidence = tuple(evidence)
        if not new_evidence:
            raise ValueError("withdrawing an estimate needs new attributable evidence")
        prior = self.dimensions[dimension]
        dimensions = dict(self.dimensions)
        dimensions[dimension] = DimensionBelief(
            status="unknown",
            evidence=prior.evidence + new_evidence,
        )
        return BeliefState(self.teammate_id, self.schema, dimensions, self.revision + 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teammate_id": self.teammate_id,
            "schema": self.schema.to_dict(),
            "revision": self.revision,
            "dimensions": {name: self.dimensions[name].to_dict() for name in self.schema.names()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeliefState":
        schema = BeliefSchema.from_dict(data["schema"])
        if set(data["dimensions"]) != set(schema.names()):
            raise ValueError("belief dimensions must exactly match the schema")
        dimensions = {
            name: DimensionBelief.from_dict(value)
            for name, value in data["dimensions"].items()
        }
        return cls(
            teammate_id=data["teammate_id"],
            schema=schema,
            dimensions=dimensions,
            revision=data["revision"],
        )

    @classmethod
    def from_json(cls, text: str) -> "BeliefState":
        return cls.from_dict(json.loads(text))
