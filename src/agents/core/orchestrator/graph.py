"""Pure graph construction for orchestrator."""
from datetime import datetime
import asyncio
import logging
from typing import Dict, Optional, Any, List, Literal, cast, TypedDict
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import START
from langchain_core.messages import BaseMessage
from langgraph.types import Command as BaseCommand

from ...common.types import (
    AgentLike, OrchestratorState, RoutingMetadata, StreamingState,
    AgentError, AgentNotFoundError, AgentExecutionError, RouterError,
    MaxErrorsExceeded
)

from ..metrics import metrics, NodeMetrics
from .router import create_router, should_continue

logger = logging.getLogger(__name__)

class NodeConfig(BaseModel):
    """Configuration for graph nodes."""
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_metrics: bool = True

class GraphNode(ABC):
    """Base class for graph nodes."""
    def __init__(self, config: NodeConfig):
        self.config = config
        self.node_metrics = metrics.start_node(self.__class__.__name__) if config.enable_metrics else None
        
    async def execute(self, state: OrchestratorState) -> Dict[str, Any]:
        """Execute node with error handling."""
        try:
            if self.node_metrics:
                self.node_metrics.start_time = datetime.now()
                
            result = await self._execute(state)
            
            if self.node_metrics:
                self.node_metrics.complete(success=True)
            return result
        except Exception as e:
            if self.node_metrics:
                self.node_metrics.complete(success=False)
                
            logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            return await self._handle_error(state, e)
            
    @abstractmethod
    async def _execute(self, state: OrchestratorState) -> Dict[str, Any]:
        """Implement actual node logic."""
        pass
        
    async def _handle_error(self, state: OrchestratorState, error: Exception) -> Dict[str, Any]:
        """Handle node execution errors."""
        updated_routing = state.routing.add_error(
            error=str(error),
            error_type="AgentExecutionError",
            agent=state.routing.current_agent,
            details={"phase": "execution"}
        )
        updated_state = state.model_copy(update={"routing": updated_routing})
        
        if self.node_metrics and updated_state.routing.error_count > self.config.max_retries:
            logger.error("Max errors exceeded, routing to error recovery")
            return {
                "next": "error_recovery",
                "state": updated_state
            }
        return {
            "next": "router",
            "state": updated_state
        }

class RouterNode(GraphNode):
    """Node for routing decisions."""
    def __init__(self, router: Any, config: NodeConfig):
        super().__init__(config)
        self.router = router
        
    async def _execute(self, state: OrchestratorState) -> Dict[str, Any]:
        """Execute routing logic."""
        logger.debug("RouterNode executing with state: %s", state)
        start_time = datetime.now()
        logger.info("Getting routing decision")
        
        try:
            # Get routing decision
            result = await asyncio.wait_for(
                self.router.route(state, {}),
                timeout=self.config.timeout_seconds
            )
            
            # Record routing metrics
            if self.config.enable_metrics and state.routing.decisions:
                routing_time = (datetime.now() - start_time).total_seconds()
                last_decision = state.routing.decisions[-1]
                metrics.add_router_decision(
                    agent=last_decision.get("next", "unknown"),
                    confidence=last_decision.get("confidence", 0.0),
                    routing_time=routing_time,
                    is_fallback=state.routing.fallback_count > 0
                )
            
            logger.debug("Routing decision: %s", result)
            logger.info("Routing decision complete")
            
            try:
                # Extract and preserve intended target
                intended_target = result.goto
                routing_data = result.update.get("routing", {})
                streaming_data = result.update.get("streaming", {})
                
                logger.debug("Processing router result", extra={
                    "goto": intended_target,
                    "routing_data": routing_data
                })

                # Ensure routing data has required fields
                if not isinstance(routing_data, dict):
                    routing_data = state.routing.model_dump()
                routing_data["current_agent"] = intended_target
                
                # Create state data with preserved target
                state_data = {
                    "messages": state.messages,
                    "next": intended_target,
                    "routing": routing_data,
                    "streaming": streaming_data or state.streaming.model_dump(),
                    "tool_state": state.tool_state,
                    "agent_ids": state.agent_ids,
                    "next_agent": intended_target,
                    "schema_version": state.schema_version
                }
                
                logger.debug("Creating state with data", extra={"state_data": state_data})
                
                # Use direct model validation
                updated_state = OrchestratorState.model_validate(state_data)
                
                logger.debug("State update complete", extra={
                    "next": updated_state.next,
                    "current_agent": updated_state.routing.current_agent
                })
                
                return {
                    "next": intended_target,
                    "state": updated_state
                }
                
            except Exception as e:
                logger.error(f"State validation error: {e}", extra={
                    "error": str(e),
                    "intended_target": intended_target
                })
                # Create error state but preserve intended target
                error_state = state.add_error(
                    str(e),
                    "ValidationError",
                    intended_target,
                    {"phase": "state_validation"}
                )
                return {
                    "next": intended_target,  # Still try to route to intended target
                    "state": error_state
                }
                
        except Exception as e:
            logger.error(f"Router execution error: {e}")
            error_state = state.add_error(
                str(e),
                "RouterError",
                None,
                {"phase": "execution"}
            )
            return {
                "next": "error_recovery",
                "state": error_state
            }

class AgentExecutorNode(GraphNode):
    """Node for executing agent actions."""
    def __init__(self, agents: Dict[str, AgentLike], config: NodeConfig):
        super().__init__(config)
        self.agents = agents
        
    async def _execute(self, state: OrchestratorState) -> Dict[str, Any]:
        """Execute selected agent."""
        current_agent_name = state.routing.current_agent or state.next
        logger.info(f"AgentExecutorNode executing with agent: {current_agent_name}")
        
        # Check if we should finish
        if current_agent_name == "FINISH" or state.next == END:
            return {"next": END, "state": state}

        # Validate agent exists
        if current_agent_name not in self.agents:
            # Try fallback to chatbot if agent not found
            if "chatbot" in self.agents:
                logger.warning(f"Agent {current_agent_name} not found, falling back to chatbot")
                current_agent_name = "chatbot"
                # Update state with fallback
                state = state.model_copy(update={
                    "routing": state.routing.model_copy(update={
                        "current_agent": "chatbot",
                        "fallback_count": state.routing.fallback_count + 1
                    })
                })
            else:
                logger.error(f"Agent {current_agent_name} not found and no fallback available")
                updated_routing = state.routing.add_error(
                    error=f"Agent {current_agent_name} not found",
                    error_type="AgentNotFoundError",
                    agent=current_agent_name
                )
                error_state = state.model_copy(update={"routing": updated_routing})
                return {
                    "next": "error_recovery",
                    "state": error_state
                }

        agent = self.agents[current_agent_name]
        logger.info(f"Found agent {current_agent_name}, executing...")
        
        # Ensure state has proper routing information
        if not state.routing.current_agent:
            state = state.model_copy(update={
                "routing": state.routing.model_copy(update={
                    "current_agent": current_agent_name
                })
            })
        
        try:
            # Shield agent invocation from cancellation
            result = await asyncio.wait_for(
                asyncio.shield(agent.ainvoke(state)),
                timeout=self.config.timeout_seconds
            )
            logger.info(f"Agent {current_agent_name} execution completed")
            
            # Validate agent result
            if not isinstance(result, dict):
                logger.error(f"Invalid agent result type: {type(result)}")
                # Update routing with error
                updated_routing = state.routing.add_error(
                    error=f"Invalid agent result type: {type(result)}",
                    error_type="ValidationError",
                    agent=current_agent_name
                )
                return {
                    "next": "error_recovery",
                    "state": state.model_copy(update={"routing": updated_routing})
                }
            
            # Ensure result has proper next state
            if "next" not in result:
                result["next"] = "FINISH" if result.get("messages", []) else "router"
            
            # Return result with preserved routing
            result["state"] = result.get("state", state)
            return result
            
        except asyncio.TimeoutError:
            logger.warning("Agent execution timed out")
            return {
                "next": "error_recovery",
                "state": state
            }
        except asyncio.CancelledError:
            logger.warning("Agent execution cancelled")
            # On cancellation, preserve the current agent and state
            return {
                "next": "router",
                "state": state.model_copy(update={
                    "routing": state.routing.model_copy(update={"error_count": 0})
                })
            }
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            updated_routing = state.routing.add_error(
                error=str(e),
                error_type="ExecutionError",
                agent=current_agent_name
            )
            return {
                "next": "error_recovery",
                "state": state.model_copy(update={"routing": updated_routing})
            }

class StreamingNode(GraphNode):
    """Node for handling streaming responses."""
    async def _execute(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle streaming state."""
        logger.debug("StreamingNode executing with state: %s", state)
        
        if not state.streaming.is_streaming:
            # Clean up any remaining streaming state
            updated_state = OrchestratorState(
                **state.model_dump(),
                streaming=StreamingState(is_streaming=False, current_buffer=None, buffers={})
            )
            logger.info("Cleaned up streaming state")
            
            return {
                "next": "router",
                "state": updated_state
            }
            
        buffer = state.streaming.current_buffer
        if buffer and buffer.is_complete:
            # Record streaming metrics
            if self.config.enable_metrics:
                metrics.add_stream_metrics(
                    stream_time=self.node_metrics.execution_time if self.node_metrics else 0,
                    token_count=len(buffer.content),
                    success=not bool(buffer.error)
                )
                
            # Create new state with ended stream
            updated_state = state.end_stream()
            
            # Create completion decision
            completion_decision = {
                "next": "FINISH",
                "confidence": 1.0,
                "reasoning": "Streaming complete",
                "capabilities_matched": [],
                "fallback_agents": []
            }
            
            # Update routing with completion decision
            final_state = updated_state.update_routing(completion_decision)
            
            return {
                "next": "router",
                "state": final_state
            }
            
        logger.info("Continuing stream processing")
        return {
            "next": "executor",
            "state": state
        }

class ErrorRecoveryNode(GraphNode):
    """Node for handling errors."""
    async def _execute(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle error recovery."""
        # Check for max retries exceeded
        if len(state.routing.errors) >= self.config.max_retries:
            logger.error("Maximum error retries reached; aborting further recursion.")
            return {"next": "abort", "state": state}

        # Log detailed error information for debugging
        latest_error = None
        if state.routing.errors:
            latest_error = state.routing.errors[-1]
            logger.error(
                f"Error recovery triggered:\n"
                f"  Type: {latest_error.error_type}\n"
                f"  Agent: {latest_error.agent or 'None'}\n"
                f"  Message: {latest_error.message}\n"
                f"  Details: {latest_error.details or 'None'}\n"
                f"  Fatal: {latest_error.is_fatal}\n"
                f"  Error Count: {state.routing.error_count}"
            )
            
            # Log additional context if available
            if latest_error.details:
                logger.debug(
                    f"Error details for {latest_error.error_type}:\n"
                    f"{latest_error.details}"
                )
        
        # Handle agent not found error specifically
        if latest_error and latest_error.error_type == "AgentNotFoundError":
            # Create clean state with default fallback
            new_state = OrchestratorState(
                messages=state.messages,
                next="chatbot",  # Default fallback agent
                routing=RoutingMetadata(
                    current_agent="chatbot",
                    decisions=[],
                    error_count=0  # Reset error count
                ),
                streaming=StreamingState(),
                tool_state=state.tool_state,
                schema_version=state.schema_version
            )
            return {"next": "executor", "state": new_state}
        elif state.routing.error_count > self.config.max_retries:
            # Switch to fallback agent
            fallback_decision = {
                "next": "chatbot",
                "confidence": 0.5,
                "reasoning": f"Error recovery fallback after {state.routing.error_count} errors",
                "capabilities_matched": [],
                "fallback_agents": []
            }
            logger.warning(
                f"Max retries ({self.config.max_retries}) exceeded:\n"
                f"  Current Agent: {state.routing.current_agent}\n"
                f"  Error Count: {state.routing.error_count}\n"
                f"Switching to fallback agent: chatbot"
            )
            
            # Create clean state with fallback
            new_state = OrchestratorState(
                messages=state.messages,
                next="chatbot",
                routing=RoutingMetadata(
                    current_agent="chatbot",
                    decisions=[fallback_decision],
                    error_count=0  # Reset error count for fresh start
                ),
                streaming=StreamingState(),  # Fresh streaming state
                tool_state=state.tool_state,
                agent_ids=state.agent_ids,
                schema_version=state.schema_version
            )
            return {
                "next": "executor",  # Go directly to executor with clean state
                "state": new_state
            }
        
        logger.info(f"Error count ({state.routing.error_count}) within limits, returning to router for retry")
        return {"next": "router", "state": state}

class OrchestratorGraph:
    """Builder for orchestrator graph."""
    def __init__(self, config: NodeConfig):
        self.config = config
        self.graph = StateGraph(OrchestratorState)
        self.graph.add_edge(START, "router")
        
    def add_router(self, router: Any) -> None:
        """Add routing node."""
        node = RouterNode(router, self.config)
        self.graph.add_node("router", node.execute)
        
    def add_executor(self, agents: Dict[str, AgentLike]) -> None:
        """Add agent executor node."""
        node = AgentExecutorNode(agents, self.config)
        self.graph.add_node("executor", node.execute)
        
    def add_streaming(self) -> None:
        """Add streaming handler node."""
        node = StreamingNode(self.config)
        self.graph.add_node("stream_handler", node.execute)
        
    def add_error_recovery(self) -> None:
        """Add error recovery node."""
        node = ErrorRecoveryNode(self.config)
        self.graph.add_node("error_recovery", node.execute)
        
    def add_edges(self) -> None:
        """Add graph edges."""
        # Router edges
        self.graph.add_conditional_edges(
            "router",
            should_continue,
            {
                "continue": "executor",
                "end": END,
                "error_recovery": "error_recovery"
            }
        )
        
        # Executor edges
        self.graph.add_conditional_edges(
            "executor",
            lambda state: (
                "end" if state.next == END or state.next == "FINISH"
                else "stream_handler" if state.streaming.is_streaming
                else "error_recovery" if state.routing.error_count > self.config.max_retries
                else "router" if state.next == "cancel"
                else "router"
            ),
            {
                "stream_handler": "stream_handler",
                "error_recovery": "error_recovery",
                "end": END,
                "router": "router"
            }
        )

        # Stream handler edges with explicit termination
        self.graph.add_conditional_edges(
            "stream_handler",
            lambda state: (
                "end" if state.next == END or state.next == "FINISH"
                else "error_recovery" if state.routing.error_count > self.config.max_retries
                else "router" if not state.streaming.is_streaming
                else "executor"
            ),
            {
                "end": END,
                "error_recovery": "error_recovery",
                "router": "router",
                "executor": "executor"
            }
        )
        
        # Error recovery edges
        self.graph.add_edge("error_recovery", "router")
        
    def build(self) -> StateGraph:
        """Build final graph."""
        logger.info("Orchestrator graph built successfully")
        return self.graph.compile()

def build_orchestrator(
    config: Optional[NodeConfig] = None,
    base_agents: Optional[Dict[str, AgentLike]] = None
) -> StateGraph:
    """Create orchestrator graph."""
    config = config or NodeConfig()
    base_agents = base_agents or {}
    
    # Create graph builder
    builder = OrchestratorGraph(config)
    
    # Create router
    router = create_router(agents=base_agents)
    
    # Add nodes
    builder.add_router(router)
    builder.add_executor(base_agents)
    builder.add_streaming()
    builder.add_error_recovery()
    
    # Add edges
    builder.add_edges()
    
    return builder.build()
