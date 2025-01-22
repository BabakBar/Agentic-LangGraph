"""Core types for agent system using Pydantic models."""
from typing import Protocol, runtime_checkable, Any, Set
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class AgentError(Exception):
    """Base class for agent-related errors."""
    pass


class CapabilityError(AgentError):
    """Raised when agent capabilities don't match requirements."""
    pass


class RegistrationError(AgentError):
    """Raised for agent registration issues."""
    pass


class RoutingError(AgentError):
    """Raised for routing-related issues."""
    pass


@runtime_checkable
class AgentLike(Protocol):
    """Protocol defining required agent interface."""
    @property
    def description(self) -> str: ...
    
    @property
    def capabilities(self) -> Set[str]: ...
    
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]: ...


class AgentMetadata(BaseModel):
    """Serializable agent metadata."""
    id: str
    description: str
    capabilities: list[str] = Field(default_factory=list)


class OrchestratorState(BaseModel):
    """Serializable state for orchestrator."""
    messages: list[BaseMessage]
    agent_ids: list[str] = Field(default_factory=list)
    next_agent: str | None = None
    
    model_config = {
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "examples": [{
                "messages": [],
                "agent_ids": ["research", "calculator"],
                "next_agent": "research"
            }]
        }
    }