"""Type definitions for tool system."""
from typing import Any, Protocol, Set, Dict
from pydantic import BaseModel

class ToolProtocol(Protocol):
    """Protocol defining required tool interface."""
    @property
    def name(self) -> str:
        """Name of the tool."""
        ...

    @property
    def description(self) -> str:
        """Description of what the tool does."""
        ...

    @property
    def capabilities(self) -> Set[str]:
        """Set of capabilities this tool provides."""
        ...

    async def execute(self, **kwargs: Dict[str, Any]) -> Any:
        """Execute the tool's functionality."""
        ...

class ToolMetadata(BaseModel):
    """Metadata for tool registration."""
    id: str
    name: str
    description: str
    capabilities: list[str]
    category: str = "core"  # Default to core category