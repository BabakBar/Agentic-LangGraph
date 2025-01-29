"""Base registry for managing runtime components with metadata."""
from typing import Any, Dict, Set, TypeVar, Generic
from pydantic import BaseModel, PrivateAttr

T = TypeVar('T')  # Type for registered items (Agent or Tool)
M = TypeVar('M')  # Type for metadata (AgentMetadata or ToolMetadata)

class RegistryError(Exception):
    """Base error for registry operations."""
    pass

class RegistryBase(BaseModel, Generic[T, M]):
    """Base registry for managing runtime components with metadata."""
    metadata: Dict[str, M] = {}
    _instances: Dict[str, T] = PrivateAttr(default_factory=dict)
    _categories: Dict[str, Set[str]] = PrivateAttr(default_factory=dict)

    def _validate_item(self, item: T) -> None:
        """Validate item implements required interface."""
        raise NotImplementedError("Subclasses must implement _validate_item")

    def _register(self, item_id: str, item: T) -> None:
        """Internal method to register an item."""
        raise NotImplementedError("Subclasses must implement _register")

    def get_item(self, item_id: str) -> T:
        """Get runtime instance by ID."""
        if item_id not in self._instances:
            raise KeyError(f"Item not found: {item_id}")
        return self._instances[item_id]

    def list_items(self) -> list[str]:
        """Get list of registered IDs."""
        return list(self.metadata.keys())

    def has_item(self, item_id: str) -> bool:
        """Check if item exists."""
        return item_id in self.metadata

    def get_by_category(self, category: str) -> Dict[str, T]:
        """Get items by category."""
        ids = self._categories.get(category, set())
        return {id: self._instances[id] for id in ids}

    def get_capabilities(self, item_id: str) -> Set[str]:
        """Get item capabilities."""
        if not self.has_item(item_id):
            raise KeyError(f"Item not found: {item_id}")
        return self._instances[item_id].capabilities