"""State management for the orchestrator agent."""
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List, Literal, Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import BaseMessage
from ...common.types import (
    AgentError, AgentNotFoundError, AgentExecutionError,
    RouterError, MaxErrorsExceeded
)

# Constants
MAX_ERRORS = 3
CURRENT_VERSION = "2.0"

logger = logging.getLogger(__name__)

class ErrorState(BaseModel):
    """Track error information."""
    message: str
    error_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    is_fatal: bool = False

class RoutingMetadata(BaseModel):
    """Track routing decisions and performance."""
    current_agent: Optional[str] = None
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[ErrorState] = Field(default_factory=list)
    fallback_count: int = 0
    error_count: int = 0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    
    def add_decision(self, decision: Dict[str, Any]) -> "RoutingMetadata":
        """Add a routing decision and return new instance."""
        return RoutingMetadata(
            current_agent=decision.get("next"),
            decisions=self.decisions + [decision],
            errors=self.errors,
            error_count=self.error_count,
            fallback_count=self.fallback_count,
            start_time=self.start_time
        )

    def add_error(self, error: str, error_type: str, agent: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> "RoutingMetadata":
        """Add an error and return new instance."""
        new_error = ErrorState(
            message=error,
            error_type=error_type,
            agent=agent,
            details=details
        )
        
        # Increment error count
        new_error_count = self.error_count + 1
        
        # Create a new instance of RoutingMetadata with the updated error count and errors
        return RoutingMetadata(
            current_agent=self.current_agent,
            decisions=self.decisions,
            errors=self.errors + [new_error],
            error_count=new_error_count,
            fallback_count=self.fallback_count,
            start_time=self.start_time
        )

class StreamBuffer(BaseModel):
    """Manage streaming tokens."""
    tokens: List[str] = Field(default_factory=list)
    is_complete: bool = False
    error: Optional[str] = None
    
    def add_token(self, token: str) -> None:
        """Add a token to the buffer."""
        self.tokens.append(token)
        
    def get_content(self) -> str:
        """Get complete buffered content."""
        return "".join(self.tokens)
        
    def clear(self) -> None:
        """Clear the buffer."""
        self.tokens = []
        self.is_complete = False
        self.error = None

class StreamingState(BaseModel):
    """Track streaming status across agents."""
    is_streaming: bool = False
    current_buffer: Optional[StreamBuffer] = None
    buffers: Dict[str, StreamBuffer] = Field(default_factory=dict)
    
    def start_stream(self, agent_id: str) -> None:
        """Start streaming for an agent."""
        self.is_streaming = True
        self.current_buffer = StreamBuffer()
        self.buffers[agent_id] = self.current_buffer
        
    def end_stream(self) -> None:
        """End current stream."""
        if self.current_buffer:
            self.current_buffer.is_complete = True
        self.is_streaming = False
        self.current_buffer = None

class ToolExecution(BaseModel):
    """Track individual tool execution."""
    tool_name: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: Literal["running", "complete", "error"] = "running"
    result: Optional[Any] = None
    error: Optional[str] = None

class ToolState(BaseModel):
    """Manage tool executions."""
    current_tools: Dict[str, ToolExecution] = Field(default_factory=dict)
    completed_tools: List[ToolExecution] = Field(default_factory=list)
    
    def start_tool(self, tool_name: str) -> str:
        """Start a tool execution."""
        execution_id = str(uuid4())
        self.current_tools[execution_id] = ToolExecution(tool_name=tool_name)
        return execution_id
        
    def complete_tool(self, execution_id: str, result: Any) -> None:
        """Complete a tool execution."""
        if execution_id not in self.current_tools:
            raise ValueError(f"No tool execution found for {execution_id}")
            
        execution = self.current_tools[execution_id]
        execution.status = "complete"
        execution.result = result
        execution.end_time = datetime.utcnow()
        
        self.completed_tools.append(execution)
        del self.current_tools[execution_id]

class OrchestratorState(BaseModel):
    """Complete orchestrator state."""
    # Core state
    messages: List[BaseMessage] = Field(...)

    # Command state
    next: Optional[str] = None
    
    # Component states
    routing: RoutingMetadata = Field(default_factory=RoutingMetadata)
    streaming: StreamingState = Field(default_factory=StreamingState)
    tools: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = CURRENT_VERSION
    
    # Backward compatibility
    agent_ids: List[str] = Field(default_factory=list)
    next_agent: Optional[str] = None
    
    @model_validator(mode="after")
    def validate_state(self) -> "OrchestratorState":
        """Validate state consistency."""
        # Validate routing state
        if self.routing.current_agent:
            if not self.routing.decisions:
                raise ValueError("Current agent set without routing decision")
            last_decision = self.routing.decisions[-1]
            if last_decision["next"] != self.routing.current_agent:
                raise ValueError("Current agent doesn't match last decision")
                
        # Validate streaming state
        if self.streaming.is_streaming:
            if not self.streaming.current_buffer:
                raise ValueError("Streaming active but no current buffer")
            if self.routing.current_agent and self.routing.current_agent not in self.streaming.buffers:
                raise ValueError("No buffer for current streaming agent")
                
        # Validate tool state
        if self.tools and not isinstance(self.tools, dict):
            raise ValueError("Tool state must be a dictionary")
            
        return self
    
    def add_error(self, error: str, error_type: str, agent: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> "OrchestratorState":
        """Add error and return new state."""
        # Update routing with error
        new_routing = self.routing.add_error(error, error_type, agent, details)
        
        return OrchestratorState(
            messages=self.messages,
            next="error_recovery",  # Route to error recovery
            routing=new_routing,
            streaming=self.streaming,
            tools=self.tools,
            agent_ids=self.agent_ids,
            next_agent=None,  # Clear current agent during error
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )
    
    def update_routing(self, decision: Dict[str, Any]) -> "OrchestratorState":
        """Update routing state."""
        # Create new RoutingMetadata instance with updated values
        new_routing = self.routing.add_decision(decision)
        
        # Create new state instance with updated routing
        return OrchestratorState(
            messages=self.messages,
            next=decision.get("next"),  # Update Command state
            routing=new_routing,
            streaming=self.streaming,
            tools=self.tools,
            agent_ids=self.agent_ids,
            next_agent=decision.get("next"),
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )

def create_initial_state(messages: list[BaseMessage]) -> OrchestratorState:
    """Create initial state for orchestrator."""
    logger.debug(f"Creating initial state with messages: {messages}")
    return OrchestratorState(
        messages=messages,
        routing=RoutingMetadata(
            next=None,
            current_agent=None,
            decisions=[]
        ),
        streaming=StreamingState(
            is_streaming=False,
            buffers={}
        ),
        tools={},
        schema_version=CURRENT_VERSION,
    )

def migrate_state(state: Dict[str, Any], target_version: str = CURRENT_VERSION) -> OrchestratorState:
    """Migrate state to target version."""
    current_version = state.get("schema_version", "1.0")
    
    if current_version == target_version:
        return OrchestratorState.model_validate(state)
        
    # Migrate from 1.0 to 2.0
    if current_version == "1.0" and target_version == "2.0":
        return OrchestratorState(
            messages=state["messages"],
            routing=RoutingMetadata(
                next=state.get("next"),
                current_agent=state.get("next"),
                decisions=[]
            ),
            streaming=StreamingState(),
            tools={},
            schema_version="2.0"
        )
        
    raise ValueError(f"Unsupported migration: {current_version} -> {target_version}")