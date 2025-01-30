"""State management for the orchestrator agent."""
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import BaseMessage

# Constants
MAX_ERRORS = 3
CURRENT_VERSION = "2.0"

class MaxErrorsExceeded(Exception):
    """Raised when max errors threshold is exceeded."""
    pass

class RoutingMetadata(BaseModel):
    """Track routing decisions and performance."""
    current_agent: Optional[str] = None
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    fallback_count: int = 0
    error_count: int = 0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    
    def add_decision(self, decision: Dict[str, Any]) -> None:
        """Add a routing decision."""
        self.decisions.append(decision)
        self.current_agent = decision["next_agent"]
        
    def add_error(self, error: str) -> None:
        """Add an error and check threshold."""
        self.error_count += 1
        if self.error_count > MAX_ERRORS:
            raise MaxErrorsExceeded()

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
            self.current_buffer.mark_complete()
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
    # Core state (immutable)
    messages: List[BaseMessage] = Field(..., frozen=True)
    
    # Component states
    routing: RoutingMetadata = Field(default_factory=RoutingMetadata)
    streaming: StreamingState = Field(default_factory=StreamingState)
    tools: ToolState = Field(default_factory=ToolState)
    
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
            if last_decision["next_agent"] != self.routing.current_agent:
                raise ValueError("Current agent doesn't match last decision")
                
        # Validate streaming state
        if self.streaming.is_streaming:
            if not self.streaming.current_buffer:
                raise ValueError("Streaming active but no current buffer")
            if self.routing.current_agent not in self.streaming.buffers:
                raise ValueError("No buffer for current streaming agent")
                
        # Validate tool state
        if any(tool.end_time is None for tool in self.tools.completed_tools):
            raise ValueError("Found completed tool without end time")
            
        return self
    
    def update_routing(self, decision: Dict[str, Any]) -> "OrchestratorState":
        """Update routing state."""
        self.routing.add_decision(decision)
        self.next_agent = decision["next_agent"]  # For backward compatibility
        self.updated_at = datetime.utcnow()
        return self
        
    def update_streaming(self, token: str) -> "OrchestratorState":
        """Update streaming state."""
        if not self.streaming.is_streaming:
            self.streaming.start_stream(self.routing.current_agent)
        self.streaming.current_buffer.add_token(token)
        self.updated_at = datetime.utcnow()
        return self
        
    def start_tool(self, tool_name: str) -> tuple["OrchestratorState", str]:
        """Start tool execution."""
        execution_id = self.tools.start_tool(tool_name)
        self.updated_at = datetime.utcnow()
        return self, execution_id
        
    def complete_tool(self, execution_id: str, result: Any) -> "OrchestratorState":
        """Complete tool execution."""
        self.tools.complete_tool(execution_id, result)
        self.updated_at = datetime.utcnow()
        return self

def create_initial_state(messages: list[BaseMessage]) -> OrchestratorState:
    """Create initial state for orchestrator."""
    return OrchestratorState(
        messages=messages,
        schema_version=CURRENT_VERSION
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
                current_agent=state.get("next_agent"),
                decisions=[]
            ),
            streaming=StreamingState(),
            tools=ToolState(),
            schema_version="2.0"
        )
        
    raise ValueError(f"Unsupported migration: {current_version} -> {target_version}")