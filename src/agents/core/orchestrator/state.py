"""State management for the orchestrator agent."""
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List, Literal, Annotated, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator, ConfigDict
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to validated dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create validated dictionary from input data."""
        return cls.model_validate(data).model_dump()
    
    def add_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Add a routing decision and return new instance."""
        return self.model_validate({
            "current_agent": decision.get("next"),
            "decisions": self.decisions + [decision],
            "errors": self.errors,
            "error_count": self.error_count,
            "fallback_count": self.fallback_count,
            "start_time": self.start_time
        }).to_dict()

    def add_error(self, error: str, error_type: str, agent: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        return self.model_validate({
            "current_agent": self.current_agent,
            "decisions": self.decisions,
            "errors": self.errors + [new_error],
            "error_count": new_error_count,
            "fallback_count": self.fallback_count,
            "start_time": self.start_time
        }).to_dict()

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to validated dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create validated dictionary from input data."""
        return cls.model_validate(data).model_dump()
    
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
    """Complete orchestrator state with immutable updates."""
    model_config = ConfigDict(frozen=True, validate_assignment=True)
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Override model_dump to ensure nested fields are dictionaries."""
        data = super().model_dump(*args, **kwargs)
        return data  # Let the validator handle conversion
    
    @classmethod
    def debug_state(cls, data: Any, context: str = "") -> None:
        """Debug helper to log state details."""
        logger.debug(f"=== State Debug [{context}] ===")
        logger.debug(f"Data type: {type(data)}")
        if isinstance(data, dict):
            for key, value in data.items():
                logger.debug(f"Field '{key}': type={type(value)}, value={value}")
        logger.debug("=== End State Debug ===")
    
    @model_validator(mode="before")
    @classmethod
    def _convert_nested(cls, data: Any) -> Any:
        """Pre-convert nested fields to dictionaries."""
        context = {"operation": "convert_nested"}
        logger.debug("Starting nested field conversion", extra={
            **context,
            "input_type": str(type(data))
        })
        
        if isinstance(data, dict):
            # Convert routing field
            if "routing" in data:
                routing_value = data["routing"]
                logger.debug("Processing routing field", extra={
                    **context,
                    "field": "routing",
                    "value_type": str(type(routing_value)),
                    "raw_value": str(routing_value)
                })
                try:
                    if isinstance(routing_value, RoutingMetadata):
                        data["routing"] = routing_value.model_dump()
                    else:
                        data["routing"] = RoutingMetadata.model_validate(routing_value).model_dump()
                    logger.debug("Routing conversion successful", extra=context)
                except Exception as e:
                    logger.error("Error converting routing", extra={**context, "error": str(e)})
                    data["routing"] = RoutingMetadata().to_dict()
            
            if "streaming" in data:
                streaming_value = data["streaming"]
                logger.debug("Processing streaming field", extra={
                    **context,
                    "field": "streaming",
                    "value_type": str(type(streaming_value)),
                    "raw_value": str(streaming_value)
                })
                try:
                    if isinstance(streaming_value, StreamingState):
                        data["streaming"] = streaming_value.model_dump()
                    else:
                        data["streaming"] = StreamingState.model_validate(streaming_value).model_dump()
                    logger.debug("Streaming conversion successful", extra=context)
                except Exception as e:
                    logger.error("Error converting streaming", extra={**context, "error": str(e)})
                    data["streaming"] = StreamingState().to_dict()
        logger.debug("Nested field conversion complete", extra=context)
        return data
    
    @classmethod
    def ensure_valid_dict(
        cls, 
        value: Union[Dict[str, Any], BaseModel], 
        model_class: Type[BaseModel]
    ) -> Dict[str, Any]:
        """Ensure value is a valid dictionary for nested models."""
        logger.debug(f"ensure_valid_dict input: type={type(value)}, model_class={model_class.__name__}")
        if isinstance(value, dict):
            logger.debug(f"Converting dict to {model_class.__name__}")
            return model_class.model_validate(value).model_dump()
        elif isinstance(value, model_class):
            logger.debug(f"Converting {model_class.__name__} instance to dict")
            return value.model_dump()
        elif value is None:
            # Handle None case by creating default instance
            logger.debug(f"Creating default {model_class.__name__} for None value")
            return model_class().model_dump()
        logger.debug(f"Invalid value type for {model_class.__name__}: {type(value)}")
        logger.debug(f"Value: {value}")
        raise ValueError(
            f"Invalid value type for {model_class.__name__}: {type(value)}"
        )

    @classmethod
    def validate_model_instance(cls, value: Any, model_class: Type[BaseModel]) -> bool:
        """Validate if a value is a proper model instance."""
        try:
            if isinstance(value, model_class):
                # Attempt to dump and validate to ensure complete validity
                dumped = value.model_dump()
                model_class.model_validate(dumped)
                return True
            elif isinstance(value, dict):
                # Attempt to validate dictionary against model
                model_class.model_validate(value)
                return True
            return False
        except Exception as e:
            logger.error(f"Validation error for {model_class.__name__}: {e}")
            return False
    
    """Complete orchestrator state."""
    # Core state
    messages: List[BaseMessage] = Field(default_factory=list)

    # Command state
    next: Optional[str] = None
    
    # Component states with validation
    routing: RoutingMetadata = Field(default_factory=RoutingMetadata, validate_default=True)
    streaming: StreamingState = Field(default_factory=StreamingState)
    tool_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = CURRENT_VERSION
    
    # Backward compatibility
    agent_ids: List[str] = Field(default_factory=list)
    next_agent: Optional[str] = None
    
    # New attributes
    
    @model_validator(mode="after")
    def validate_state(self) -> "OrchestratorState":
        """Validate state consistency."""
        context = {
            "operation": "validate_state",
            "routing_type": str(type(self.routing)),
            "streaming_type": str(type(self.streaming)),
            "current_agent": self.routing.current_agent,
            "next": self.next,
            "routing_content": self.routing.model_dump(),
            "streaming_content": self.streaming.model_dump()
        }
        
        logger.debug("Starting state validation", extra=context)
        
        validation_errors = []
        validation_context = {**context, "validation_errors": validation_errors}
        
        # Validate routing state
        if self.routing.current_agent:
            if not self.routing.decisions:
                raise ValueError("Current agent set without routing decision")
            last_decision = self.routing.decisions[-1]
            if last_decision["next"] != self.routing.current_agent:
                raise ValueError("Current agent doesn't match last decision")
                
        # Validate model instances
        if not self.validate_model_instance(self.routing, RoutingMetadata):
            logger.error(
                f"Invalid routing state type: {type(self.routing)}, "
                f"value: {self.routing}"
            , extra=validation_context)
            raise ValueError("Invalid routing state")
            
        if not self.validate_model_instance(self.streaming, StreamingState):
            logger.error("Invalid streaming state", extra={
                **validation_context,
                "streaming_type": str(type(self.streaming))
            })
            raise ValueError("Invalid streaming state")
                
        if validation_errors:
            raise ValueError(f"State validation failed: {', '.join(validation_errors)}")
            
        # Validate streaming state
        if self.streaming.is_streaming:
            if not self.streaming.current_buffer:
                raise ValueError("Streaming active but no current buffer")
            if self.routing.current_agent and self.routing.current_agent not in self.streaming.buffers:
                raise ValueError("No buffer for current streaming agent")
                
        # Validate tool state
        if self.tool_state and not isinstance(self.tool_state, dict):
            raise ValueError("Tool state must be a dictionary")
            
        logger.debug("State validation successful", extra=context)
        return self
    
    @classmethod
    def create_validated_state(cls, data: Dict[str, Any]) -> "OrchestratorState":
        """Create a new state instance with validated components."""
        logger.debug("Creating validated state")
        cls.debug_state(data, "Input to create_validated_state")
        try:
            # Ensure routing and streaming are proper dictionaries
            if "routing" in data:
                data["routing"] = cls.ensure_valid_dict(data["routing"], RoutingMetadata)
            if "streaming" in data:
                data["streaming"] = cls.ensure_valid_dict(data["streaming"], StreamingState)
            cls.debug_state(data, "After converting nested fields")
                
            # Create and validate new state
            new_state = cls.model_validate(data)
            logger.debug(f"Created new state: {new_state}")
            return new_state
        except Exception as e:
            logger.error(f"Error creating validated state: {e}")
            raise ValueError(f"Failed to create valid state: {e}")
    
    def add_error(self, error: str, error_type: str, agent: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> "OrchestratorState":
        """Add error and return new state."""
        logger.debug(f"Adding error: type={error_type}, agent={agent}, error={error}")
        # Convert existing routing to dict first
        routing_dict = self.routing.model_dump()
        new_routing = RoutingMetadata.model_validate(routing_dict).add_error(error, error_type, agent, details)
        logger.debug(f"Created new routing with error: {new_routing}")

        return OrchestratorState(
            messages=self.messages,
            next="error_recovery",  # Route to error recovery
            routing=new_routing.model_dump(),  # Convert back to dict
            streaming=self.streaming.model_dump(),  # Convert to dict
            tool_state=self.tool_state,
            agent_ids=self.agent_ids,
            next_agent=None,  # Clear current agent during error
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )
    
    def update_routing(self, decision: Dict[str, Any]) -> "OrchestratorState":
        """Update routing state."""
        # Create new RoutingMetadata instance with updated values
        logger.debug(f"Updating routing with decision: {decision}")
        logger.debug(f"Current routing state type: {type(self.routing)}")
        # Convert existing routing to dict first
        routing_dict = self.routing.model_dump()
        new_routing = RoutingMetadata.model_validate(routing_dict).add_decision(decision)
        logger.debug(f"New routing state: {new_routing}")
        
        # Create new state instance with updated routing
        return OrchestratorState(
            messages=self.messages,
            next=decision.get("next"),  # Update Command state
            routing=new_routing.model_dump(),  # Convert back to dict
            streaming=self.streaming.model_dump(),  # Convert to dict
            tool_state=self.tool_state,
            agent_ids=self.agent_ids,
            next_agent=decision.get("next"),
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )

def create_initial_state(messages: list[BaseMessage], agent_ids: list[str] = None) -> OrchestratorState:
    """Create initial state for orchestrator."""
    context = {
        "operation": "create_initial_state",
        "message_count": len(messages),
        "agent_ids": agent_ids
    }
    logger.debug("Creating initial state", extra=context)
    
    # Create initial routing state as dict
    initial_routing = RoutingMetadata(
        current_agent=None,
        decisions=[],
        errors=[],
        error_count=0,
        fallback_count=0
    )
    routing_dict = initial_routing.model_dump()
    logger.debug("Created initial routing", extra={
        **context,
        "routing_type": str(type(initial_routing)),
        "routing_dict": routing_dict
    })
    
    # Create initial streaming state as dict
    initial_streaming = StreamingState()
    streaming_dict = initial_streaming.model_dump()
    logger.debug("Created initial streaming", extra={
        **context,
        "streaming_type": str(type(initial_streaming)),
        "streaming_dict": streaming_dict
    })
    
    state = OrchestratorState(
        agent_ids=agent_ids if agent_ids is not None else [],
        messages=messages,
        routing=routing_dict,
        streaming=streaming_dict,
        tool_state={},
        schema_version=CURRENT_VERSION,
    )
    
    state_dict = state.model_dump()
    logger.debug("Created initial state", extra={
        **context,
        "state_type": str(type(state)),
        "state_dict": state_dict
    })
    
    return state


def migrate_state(state: Dict[str, Any], target_version: str = CURRENT_VERSION) -> OrchestratorState:
    """Migrate state to target version."""
    current_version = state.get("schema_version", "1.0")
    
    if current_version == target_version:
        return OrchestratorState.model_validate(state)
        
    logger.debug(f"Migrating state from version {current_version} to {target_version}")
    # Migrate from 1.0 to 2.0
    if current_version == "1.0" and target_version == "2.0":
        return OrchestratorState(
            messages=state["messages"],
            routing=OrchestratorState.ensure_valid_dict({
                "next": state.get("next"),
                "current_agent": state.get("next"),
                "decisions": []
            }, RoutingMetadata),
            streaming=OrchestratorState.ensure_valid_dict(StreamingState(), StreamingState),
            tool_state={},
            schema_version="2.0"
        )
        
    error_msg = f"Unsupported migration: {current_version} -> {target_version}"
    raise ValueError(error_msg)