"""Tool registry implementation."""
from typing import Dict, Set

from ..core.registry_base import RegistryBase, RegistryError
from .tool_types import ToolProtocol, ToolMetadata

class ToolRegistryError(RegistryError):
    """Specific error type for tool registry operations."""
    pass

class ToolRegistry(RegistryBase[ToolProtocol, ToolMetadata]):
    """Registry managing tool metadata and runtime instances."""

    def _validate_item(self, tool: ToolProtocol) -> None:
        """Validate tool implements required interface."""
        # Protocol validation is handled by type checking
        # Additional runtime checks can be added here if needed
        required_attrs = ['name', 'description', 'capabilities', 'execute']
        missing_attrs = [attr for attr in required_attrs if not hasattr(tool, attr)]
        if missing_attrs:
            raise ToolRegistryError(
                f"Tool missing required attributes: {', '.join(missing_attrs)}"
            )

    def _register(self, tool_id: str, tool: ToolProtocol) -> None:
        """Internal method to register a tool."""
        self.metadata[tool_id] = ToolMetadata(
            id=tool_id,
            name=tool.name,
            description=tool.description,
            capabilities=list(tool.capabilities)
        )
        self._instances[tool_id] = tool

    def register_core_tool(self, tool_id: str, tool: ToolProtocol) -> None:
        """Register a core tool."""
        if tool_id in self.metadata:
            raise ToolRegistryError(f"Tool already registered: {tool_id}")
        
        self._validate_item(tool)
        self._categories.setdefault("core", set()).add(tool_id)
        self._register(tool_id, tool)

    def register_specialized_tool(self, tool_id: str, tool: ToolProtocol) -> None:
        """Register a specialized tool."""
        if tool_id in self.metadata:
            raise ToolRegistryError(f"Tool already registered: {tool_id}")
        
        self._validate_item(tool)
        self._categories.setdefault("specialized", set()).add(tool_id)
        self._register(tool_id, tool)

    def get_tool(self, tool_id: str) -> ToolProtocol:
        """Get a tool by ID."""
        return self.get_item(tool_id)

    def get_core_tools(self) -> Dict[str, ToolProtocol]:
        """Get all core tools."""
        return self.get_by_category("core")

    def get_specialized_tools(self) -> Dict[str, ToolProtocol]:
        """Get all specialized tools."""
        return self.get_by_category("specialized")

    def get_tools_by_capability(self, capability: str) -> Dict[str, ToolProtocol]:
        """Get all tools that provide a specific capability."""
        return {
            tool_id: tool
            for tool_id, tool in self._instances.items()
            if capability in tool.capabilities
        }