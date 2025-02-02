"""Core types for agent system using Pydantic models."""

class AgentError(Exception):
    """Base class for agent-related errors."""
    pass

class AgentNotFoundError(AgentError):
    """Raised when an agent is not found in the registry."""
    pass

class AgentExecutionError(AgentError):
    """Raised when an agent fails during execution."""
    pass

class RouterError(AgentError):
    """Raised when routing fails."""
    pass

class MaxErrorsExceeded(AgentError):
    """Raised when max errors threshold is exceeded."""
    pass

from datetime import datetime
from typing import Protocol, runtime_checkable, Any, Set, Optional, List, Dict, Annotated, Self
from pydantic import BaseModel, Field, ConfigDict, model_validator
from langchain_core.messages import BaseMessage
from langgraph.types import Command as BaseCommand
import operator

def update_dict(old: dict, new: dict) -> dict:
    """Merge two dictionaries, new values override old ones."""
    return {**old, **new} if old and new else new or old or {}

class AgentError(Exception):
    """Base class for agent-related errors."""
    pass

class CapabilityError(AgentError):
    """Raised when agent capabilities don't match requirements."""
    pass

class RegistrationError(AgentError):
    """Raised for agent registration issues."""
    pass

class RoutingError(BaseModel):
    """Structured routing error information."""
    timestamp: datetime
    error: str
    input: str
    agent: Optional[str] = None

class MaxErrorsExceeded(AgentError):
    """Raised when maximum number of retries is exceeded."""
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
    capabilities_matched: List[str] = Field(default_factory=list)
    fallback_agents: List[str] = Field(default_factory=list)

class RoutingMetadata(BaseModel):
    """Mutable routing state information."""
    current_agent: Optional[str] = None
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[RoutingError] = Field(default_factory=list)
    fallback_count: int = 0
    error_count: int = 0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    
    def add_decision(self, decision: Dict[str, Any]) -> Self:
        """Add a routing decision and return new instance."""
        return RoutingMetadata(
            current_agent=decision.get("next"),
            decisions=self.decisions + [decision],
            errors=self.errors,
            fallback_count=self.fallback_count,
            error_count=self.error_count,
            start_time=self.start_time
        )
        
    def add_error(self, error: str, error_type: str, agent: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Self:
        """Add an error, increment error count, and return new instance."""
        new_error = RoutingError(
            timestamp=datetime.now(),
            error=error,
            input=error_type,  # Using input field to store error_type for compatibility
            agent=agent
        )
        
        new_error_count = self.error_count + 1
        
        return RoutingMetadata(
            current_agent=self.current_agent,
            decisions=self.decisions,
            errors=self.errors + [new_error],
            fallback_count=self.fallback_count,
            error_count=new_error_count,
            start_time=self.start_time
        )
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
    
    def add_token(self, token: str) -> Self:
        """Add a token to the buffer and return new instance."""
        return StreamBuffer(
            content=self.content + [token],
            is_complete=self.is_complete,
            error=self.error,
            agent_id=self.agent_id
        )
    
    def get_content(self) -> str:
        """Get complete buffered content."""
        return "".join(self.content)
    
    def mark_complete(self) -> Self:
        """Mark the buffer as complete and return new instance."""
        return StreamBuffer(
            content=self.content,
            is_complete=True,
            error=self.error,
            agent_id=self.agent_id
        )

class StreamingState(BaseModel):
    """Manages streaming state across agents."""
    is_streaming: bool = False
    current_buffer: Optional[StreamBuffer] = None
    buffers: Dict[str, StreamBuffer] = Field(default_factory=dict)
    
    def start_stream(self, agent_id: str) -> Self:
        """Start streaming for an agent and return new instance."""
        new_buffer = StreamBuffer(agent_id=agent_id)
        new_buffers = dict(self.buffers)
        new_buffers[agent_id] = new_buffer
        return StreamingState(
            is_streaming=True,
            current_buffer=new_buffer,
            buffers=new_buffers
        )
    
    def end_stream(self) -> Self:
        """End current stream and return new instance."""
        new_buffers = dict(self.buffers)
        if self.current_buffer:
            new_buffers[self.current_buffer.agent_id] = self.current_buffer.mark_complete()
        return StreamingState(
            is_streaming=False,
            current_buffer=None,
            buffers=new_buffers
        )
    
    def add_token(self, token: str) -> Self:
        """Add token to current buffer and return new instance."""
        if not self.current_buffer:
            return self
        new_buffer = self.current_buffer.add_token(token)
        new_buffers = dict(self.buffers)
        new_buffers[new_buffer.agent_id] = new_buffer
        return StreamingState(
            is_streaming=self.is_streaming,
            current_buffer=new_buffer,
            buffers=new_buffers
        )
    
    def set_error(self, error: str) -> Self:
        """Set error on current buffer and return new instance."""
        if not self.current_buffer:
            return self
        new_buffer = StreamBuffer(
            content=self.current_buffer.content,
            is_complete=True,
            error=error,
            agent_id=self.current_buffer.agent_id
        )
        new_buffers = dict(self.buffers)
        new_buffers[new_buffer.agent_id] = new_buffer
        return StreamingState(
            is_streaming=False,
            current_buffer=None,
            buffers=new_buffers
        )

class ToolState(BaseModel):
    """State management for tool execution."""
    tool_states: Dict[str, Any] = Field(default_factory=dict)
    last_update: Optional[datetime] = None
    
    def update(self, tool_id: str, state: Any) -> Self:
        """Update state for a specific tool and return new instance."""
        new_states = dict(self.tool_states)
        new_states[tool_id] = state
        return ToolState(
            tool_states=new_states,
            last_update=datetime.now()
        )

    def get(self, tool_id: str, default: Any = None) -> Any:
        """Get state for a specific tool."""
        return self.tool_states.get(tool_id, default)

    def clear(self, tool_id: str) -> Self:
        """Clear state for a specific tool and return new instance."""
        new_states = dict(self.tool_states)
        if tool_id in new_states:
            del new_states[tool_id]
        return ToolState(
            tool_states=new_states,
            last_update=datetime.now()
        )

class OrchestratorState(BaseModel):
    """Serializable state for orchestrator with immutable core."""
    # Immutable conversation history
    messages: List[BaseMessage] = Field(..., frozen=True)

    # Command state
    next: Optional[str] = None
    
    # Mutable routing state
    routing: RoutingMetadata = Field(default_factory=RoutingMetadata)

    # Streaming state management
    streaming: StreamingState = Field(default_factory=StreamingState)

    # Tool state management with reducer
    tool_state: Annotated[Dict[str, Any], update_dict] = Field(default_factory=dict)
    
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
    
    def update_routing(self, decision: Dict[str, Any]) -> Self:
        """Update routing state with new decision."""
        return OrchestratorState(
            messages=self.messages,
            next=decision.get("next"),
            routing=self.routing.add_decision(decision),
            streaming=self.streaming,
            tool_state=self.tool_state,
            agent_ids=self.agent_ids,
            next_agent=decision.get("next"),
            schema_version=self.schema_version
        )
    
    def add_error(self, error: str, input_text: str, agent: Optional[str] = None) -> Self:
        """Add routing error to state."""
        return OrchestratorState(
            messages=self.messages,
            next=self.next,
            routing=self.routing.add_error(error, input_text, agent),
            streaming=self.streaming,
            tool_state=self.tool_state,
            agent_ids=self.agent_ids,
            next_agent=self.next_agent,
            schema_version=self.schema_version
        )

    def update_tool_state(self, tool_id: str, state: Any) -> Self:
        """Update state for a specific tool."""
        return OrchestratorState(
            messages=self.messages,
            next=self.next,
            routing=self.routing,
            streaming=self.streaming,
            tool_state=self.tool_state.update(tool_id, state),
            agent_ids=self.agent_ids,
            next_agent=self.next_agent,
            schema_version=self.schema_version
        )

    def get_tool_state(self, tool_id: str, default: Any = None) -> Any:
        """Get state for a specific tool."""
        return self.tool_state.get(tool_id, default)

    def clear_tool_state(self, tool_id: str) -> Self:
        """Clear state for a specific tool."""
        return OrchestratorState(
            messages=self.messages,
            next=self.next,
            routing=self.routing,
            streaming=self.streaming,
            tool_state=self.tool_state.clear(tool_id),
            agent_ids=self.agent_ids,
            next_agent=self.next_agent,
            schema_version=self.schema_version
        )

    def start_stream(self, agent_id: str) -> Self:
        """Start streaming for an agent."""
        return OrchestratorState(
            messages=self.messages,
            next=self.next,
            routing=self.routing,
            streaming=self.streaming.start_stream(agent_id),
            tool_state=self.tool_state,
            agent_ids=self.agent_ids,
            next_agent=self.next_agent,
            schema_version=self.schema_version
        )

    def end_stream(self) -> Self:
        """End current stream."""
        return OrchestratorState(
            messages=self.messages,
            next=self.next,
            routing=self.routing,
            streaming=self.streaming.end_stream(),
            tool_state=self.tool_state,
            agent_ids=self.agent_ids,
            next_agent=self.next_agent,
            schema_version=self.schema_version
        )

    def add_token(self, token: str) -> Self:
        """Add token to current stream."""
        return OrchestratorState(
            messages=self.messages,
            next=self.next,
            routing=self.routing,
            streaming=self.streaming.add_token(token),
            tool_state=self.tool_state,
            agent_ids=self.agent_ids,
            next_agent=self.next_agent,
            schema_version=self.schema_version
        )

    def set_stream_error(self, error: str) -> Self:
        """Set error on current stream."""
        return OrchestratorState(
            messages=self.messages,
            next=self.next,
            routing=self.routing,
            streaming=self.streaming.set_error(error),
            tool_state=self.tool_state,
            agent_ids=self.agent_ids,
            next_agent=self.next_agent,
            schema_version=self.schema_version
        )
