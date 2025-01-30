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

class StreamBuffer(BaseModel):
    """Manages streaming content buffer."""
    content: List[str] = Field(default_factory=list)
    is_complete: bool = False
    error: Optional[str] = None
    agent_id: Optional[str] = None
    
    def add_token(self, token: str) -> None:
        """Add a token to the buffer."""
        self.content.append(token)
    
    def get_content(self) -> str:
        """Get complete buffered content."""
        return "".join(self.content)
    
    def mark_complete(self) -> None:
        """Mark the buffer as complete."""
        self.is_complete = True
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.content = []
        self.is_complete = False
        self.error = None

class StreamingState(BaseModel):
    """Manages streaming state across agents."""
    is_streaming: bool = False
    current_buffer: Optional[StreamBuffer] = None
    buffers: Dict[str, StreamBuffer] = Field(default_factory=dict)
    
    def start_stream(self, agent_id: str) -> None:
        """Start streaming for an agent."""
        self.is_streaming = True
        self.current_buffer = StreamBuffer(agent_id=agent_id)
        self.buffers[agent_id] = self.current_buffer
    
    def end_stream(self) -> None:
        """End current stream."""
        if self.current_buffer:
            self.current_buffer.mark_complete()
        self.is_streaming = False
        self.current_buffer = None
    
    def add_token(self, token: str) -> None:
        """Add token to current buffer."""
        if self.current_buffer:
            self.current_buffer.add_token(token)
    
    def set_error(self, error: str) -> None:
        """Set error on current buffer."""
        if self.current_buffer:
            self.current_buffer.error = error
            self.end_stream()

class ToolState(BaseModel):
    """State management for tool execution."""
    tool_states: Dict[str, Any] = Field(default_factory=dict)
    last_update: Optional[datetime] = None
    
    def update(self, tool_id: str, state: Any) -> "ToolState":
        """Update state for a specific tool."""
        self.tool_states[tool_id] = state
        self.last_update = datetime.now()
        return self

    def get(self, tool_id: str, default: Any = None) -> Any:
        """Get state for a specific tool."""
        return self.tool_states.get(tool_id, default)

    def clear(self, tool_id: str) -> None:
        """Clear state for a specific tool."""
        if tool_id in self.tool_states:
            del self.tool_states[tool_id]
            self.last_update = datetime.now()

class OrchestratorState(BaseModel):
    """Serializable state for orchestrator with immutable core."""
    # Immutable conversation history
    messages: List[BaseMessage] = Field(..., frozen=True)
    
    # Mutable routing state
    routing: RoutingMetadata = Field(default_factory=RoutingMetadata)

    # Streaming state management
    streaming: StreamingState = Field(default_factory=StreamingState)

    # Tool state management
    tool_state: ToolState = Field(default_factory=ToolState)
    
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

    def update_tool_state(self, tool_id: str, state: Any) -> "OrchestratorState":
        """Update state for a specific tool."""
        self.tool_state.update(tool_id, state)
        return self

    def get_tool_state(self, tool_id: str, default: Any = None) -> Any:
        """Get state for a specific tool."""
        return self.tool_state.get(tool_id, default)

    def clear_tool_state(self, tool_id: str) -> "OrchestratorState":
        """Clear state for a specific tool."""
        self.tool_state.clear(tool_id)
        return self

    def start_stream(self, agent_id: str) -> "OrchestratorState":
        """Start streaming for an agent."""
        self.streaming.start_stream(agent_id)
        return self

    def end_stream(self) -> "OrchestratorState":
        """End current stream."""
        self.streaming.end_stream()
        return self

    def add_token(self, token: str) -> "OrchestratorState":
        """Add token to current stream."""
        self.streaming.add_token(token)
        return self

    def set_stream_error(self, error: str) -> "OrchestratorState":
        """Set error on current stream."""
        self.streaming.set_error(error)
        return self
