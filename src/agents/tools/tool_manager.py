"""Tool registry and management."""
from typing import Dict, Any

from .tool_registry import ToolRegistry
from .calculator import calculator

# Initialize registry
_tool_registry = ToolRegistry()

# Register core tools
_tool_registry.register_core_tool(
    "calculator",
    calculator
)

def get_tool(tool_id: str) -> Any:
    """Get a tool by ID."""
    return _tool_registry.get_tool(tool_id)

def get_all_tools() -> Dict[str, Any]:
    """Get all registered tools."""
    tools = {}
    tools.update(_tool_registry.get_core_tools())
    tools.update(_tool_registry.get_specialized_tools())
    return tools

def get_tools_by_capability(capability: str) -> Dict[str, Any]:
    """Get all tools that provide a specific capability."""
    return _tool_registry.get_tools_by_capability(capability)