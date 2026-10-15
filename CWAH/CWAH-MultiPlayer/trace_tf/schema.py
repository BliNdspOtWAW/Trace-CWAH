"""Replaceable definitions of teammate-belief dimensions."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    description: str
    timescale: str = "unspecified"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("dimension name must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("dimension description must be a string")
        if not isinstance(self.timescale, str):
            raise TypeError("dimension timescale must be a string")

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "timescale": self.timescale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionSpec":
        return cls(
            name=data["name"],
            description=data["description"],
            timescale=data.get("timescale", "unspecified"),
        )


@dataclass(frozen=True)
class BeliefSchema:
    schema_id: str
    dimensions: Tuple[DimensionSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema_id, str) or not self.schema_id.strip():
            raise ValueError("schema_id must be a non-empty string")
        dimensions = tuple(self.dimensions)
        if not dimensions or not all(isinstance(item, DimensionSpec) for item in dimensions):
            raise ValueError("schema needs at least one DimensionSpec")
        names = [item.name for item in dimensions]
        if len(names) != len(set(names)):
            raise ValueError("dimension names must be unique")
        object.__setattr__(self, "dimensions", dimensions)

    @classmethod
    def define(cls, schema_id: str, dimensions: Iterable[DimensionSpec]) -> "BeliefSchema":
        return cls(schema_id=schema_id, dimensions=tuple(dimensions))

    def names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.dimensions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "dimensions": [item.to_dict() for item in self.dimensions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeliefSchema":
        return cls.define(
            schema_id=data["schema_id"],
            dimensions=[DimensionSpec.from_dict(item) for item in data["dimensions"]],
        )


DEFAULT_SCHEMA = BeliefSchema.define(
    "trace-v0-six-dimensions",
    (
        DimensionSpec("information", "Task facts the teammate may know, such as object locations", "fast"),
        DimensionSpec("capability", "Observed ability or efficiency on task actions", "slow"),
        DimensionSpec("intent", "Current goal, plan, or subtask inferred from evidence", "fast"),
        DimensionSpec("preference", "Observed coordination and communication preference", "slow"),
        DimensionSpec("reliability", "Agreement between teammate reports or commitments and outcomes", "slow"),
        DimensionSpec("uncertainty", "Uncertainty about the teammate not captured by a single estimate", "variable"),
    ),
)
