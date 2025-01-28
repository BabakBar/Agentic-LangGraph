"""Core types for agent system using Pydantic models."""
from datetime import datetime
from typing import Protocol, runtime_checkable, Any, Set, Optional, List, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator
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
    enabled: bool = True

class RouterDecision(BaseModel):
    """Raw routing decision from LLM or keyword-based routing."""
    next: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    reasoning: Optional[str] = None
    alternatives: List[str] = Field(default_factory=list)

class RoutingError(BaseModel):
    """Structured routing error information."""
    timestamp: datetime
    error: str
    input: str
    agent: Optional[str] = None

class RoutingMetadata(BaseModel):
    """Mutable routing state information."""
    current_agent: Optional[str] = None
    decision_history: List[RouterDecision] = Field(default_factory=list)
    fallback_used: bool = False
    errors: List[RoutingError] = Field(default_factory=list)

class ValidatedRouterOutput(RouterDecision):
    """Validated routing output with registry awareness."""
    @model_validator(mode="after")
    def validate_decision(self) -> "ValidatedRouterOutput":
        if self.confidence < 0.5:  # Minimum confidence threshold
            raise ValueError(f"Confidence too low: {self.confidence}")
        return self

class OrchestratorState(BaseModel):
    """Serializable state for orchestrator with immutable core."""
    # Immutable conversation history
    messages: List[BaseMessage] = Field(..., frozen=True)
    
    # Mutable routing state
    routing: RoutingMetadata = Field(default_factory=RoutingMetadata)
    
    # Backward compatibility
    agent_ids: List[str] = Field(default_factory=list)
    next_agent: Optional[str] = None
    
    # Version tracking
    schema_version: str = "2.0"
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [{
                "messages": [],
                "agent_ids": ["research", "calculator"],
                "next_agent": "research",
                "schema_version": "2.0"
            }]
        }
    )
    
    def update_routing(self, decision: RouterDecision) -> "OrchestratorState":
        """Update routing state with new decision."""
        self.routing.current_agent = decision.next
        self.routing.decision_history.append(decision)
        self.next_agent = decision.next  # For backward compatibility
        return self
    
    def add_error(self, error: str, input_text: str, agent: Optional[str] = None) -> "OrchestratorState":
        """Add routing error to state."""
        self.routing.errors.append(
            RoutingError(
                timestamp=datetime.now(),
                error=error,
                input=input_text,
                agent=agent
            )
        )
        return self
