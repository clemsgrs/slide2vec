"""Generic component registry for slide2vec."""

from __future__ import annotations

from typing import Any


class Registry:
    """A simple name → class registry with optional metadata.

    Used by encoders to register components by name for discovery and
    instantiation.
    """

    def __init__(self, domain: str) -> None:
        self._domain = domain
        self._entries: dict[str, _Entry] = {}

    def register(
        self, name: str, cls: type, *, metadata: dict[str, Any] | None = None
    ) -> None:
        """Register a class under a given name."""
        if name in self._entries:
            msg = f"'{name}' is already registered in the {self._domain} registry"
            raise ValueError(msg)
        self._entries[name] = _Entry(cls=cls, metadata=metadata or {})

    def register_decorator(
        self, name: str, *, metadata: dict[str, Any] | None = None
    ):
        """Decorator form of register."""

        def decorator(cls: type) -> type:
            self.register(name, cls, metadata=metadata)
            return cls

        return decorator

    def get(self, name: str) -> type:
        """Retrieve a registered class by name."""
        if name not in self._entries:
            available = ", ".join(sorted(self._entries)) or "(none)"
            msg = f"'{name}' not found in {self._domain} registry. Available: {available}"
            raise KeyError(msg)
        return self._entries[name].cls

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def list(self) -> list[str]:
        """List all registered component names."""
        return list(self._entries.keys())

    def info(self, name: str) -> dict[str, Any]:
        """Get metadata for a registered component."""
        if name not in self._entries:
            msg = f"'{name}' not found in {self._domain} registry"
            raise KeyError(msg)
        entry = self._entries[name]
        return {"name": name, **entry.metadata}

    def list_with_metadata(self) -> list[dict[str, Any]]:
        """List all components with their metadata."""
        return [self.info(name) for name in self._entries]


class _Entry:
    __slots__ = ("cls", "metadata")

    def __init__(self, cls: type, metadata: dict[str, Any]) -> None:
        self.cls = cls
        self.metadata = metadata
