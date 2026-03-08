"""Sport module registry.

Each sport (NBA, Tennis, etc.) registers as a module that provides its own
data feed, quant agent, and strategies. The registry lets main.py discover
and launch sport modules by name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sports.base import SportModule

_REGISTRY: dict[str, type[SportModule]] = {}


def register(name: str):
    """Class decorator: ``@register("nba")``."""
    def decorator(cls: type[SportModule]) -> type[SportModule]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_sport(name: str) -> SportModule:
    """Instantiate a registered sport module by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown sport {name!r}. Available: {available}")
    return cls()


def available_sports() -> list[str]:
    return sorted(_REGISTRY)
