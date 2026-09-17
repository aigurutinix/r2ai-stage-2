
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, TypeAlias, TypeVar

K = TypeVar("K", bound=Hashable)
Scalar: TypeAlias = int | float | bool


class KeyedOperationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Series(Generic[K]):
    values: dict[K, float]


@dataclass(frozen=True, slots=True)
class KeySet(Generic[K]):
    keys: frozenset[K]


@dataclass(frozen=True, slots=True)
class SelectedKey(Generic[K]):
    key: K


def filter_keys(
    series: Series[K],
    *,
    comparison: str,
    threshold: float,
    require_proper_subset: bool = True,
) -> KeySet[K]:
    predicates = {
        "gt": lambda value: value > threshold,
        "gte": lambda value: value >= threshold,
        "lt": lambda value: value < threshold,
        "lte": lambda value: value <= threshold,
    }
    predicate = predicates.get(comparison)
    if predicate is None:
        raise KeyedOperationError(f"Unsupported comparison: {comparison!r}")
    keys = frozenset(key for key, value in series.values.items() if predicate(value))
    if not keys:
        raise KeyedOperationError("filter_keys produced an empty set")
    if require_proper_subset and len(keys) == len(series.values):
        raise KeyedOperationError("filter_keys did not remove any keys")
    return KeySet(keys)


def select_key(series: Series[K], allowed_keys: KeySet[K], *, operation: str) -> SelectedKey[K]:
    missing = allowed_keys.keys - series.values.keys()
    if missing:
        raise KeyedOperationError(f"Selector has no value for keys: {sorted(missing, key=str)}")
    if not allowed_keys.keys:
        raise KeyedOperationError("select_key received an empty KeySet")
    if operation not in ("argmax", "argmin"):
        raise KeyedOperationError(f"Unsupported selector operation: {operation!r}")

    values = {key: series.values[key] for key in allowed_keys.keys}
    best_value = max(values.values()) if operation == "argmax" else min(values.values())
    winners = [key for key, value in values.items() if value == best_value]
    if len(winners) != 1:
        raise KeyedOperationError(f"Selector has no unique winner: {sorted(winners, key=str)}")
    return SelectedKey(winners[0])


def lookup(series: Series[K], selected: SelectedKey[K]) -> Scalar:
    if selected.key not in series.values:
        raise KeyedOperationError(f"Lookup is missing the selected key: {selected.key!r}")
    return series.values[selected.key]
