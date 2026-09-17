
from __future__ import annotations

from vifinqa.generation.hard.recipe.base import GraphError, Operation

__all__ = ["Operation", "register_operation", "get_operation", "registered_operations"]

_REGISTRY: dict[str, Operation] = {}


def register_operation(operation: Operation) -> None:
    if operation.name in _REGISTRY:
        raise GraphError(f"Operation is already registered: {operation.name}")
    _REGISTRY[operation.name] = operation


def get_operation(name: str) -> Operation:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise GraphError(f"Operation is not registered: {name!r}") from None


def registered_operations() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
